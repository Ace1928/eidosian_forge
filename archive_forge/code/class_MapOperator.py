import copy
import functools
import itertools
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import Any, Callable, Deque, Dict, Iterator, List, Optional, Set, Union
import ray
from ray import ObjectRef
from ray._raylet import ObjectRefGenerator
from ray.data._internal.compute import (
from ray.data._internal.execution.interfaces import (
from ray.data._internal.execution.interfaces.physical_operator import (
from ray.data._internal.execution.operators.base_physical_operator import (
from ray.data._internal.execution.operators.map_transformer import (
from ray.data._internal.stats import StatsDict
from ray.data.block import Block, BlockAccessor, BlockExecStats, BlockMetadata
from ray.data.context import DataContext
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
class MapOperator(OneToOneOperator, ABC):
    """A streaming operator that maps input bundles 1:1 to output bundles.

    This operator implements the distributed map operation, supporting both task
    and actor compute strategies.
    """

    def __init__(self, map_transformer: MapTransformer, input_op: PhysicalOperator, name: str, target_max_block_size: Optional[int], min_rows_per_bundle: Optional[int], ray_remote_args: Optional[Dict[str, Any]]):
        self._map_transformer = map_transformer
        self._ray_remote_args = _canonicalize_ray_remote_args(ray_remote_args or {})
        self._ray_remote_args_factory = None
        self._remote_args_for_metrics = copy.deepcopy(self._ray_remote_args)
        self._block_ref_bundler = _BlockRefBundler(min_rows_per_bundle)
        self._output_queue: _OutputQueue = None
        self._output_metadata: List[BlockMetadata] = []
        self._data_tasks: Dict[int, DataOpTask] = {}
        self._next_data_task_idx = 0
        self._metadata_tasks: Dict[int, MetadataOpTask] = {}
        self._next_metadata_task_idx = 0
        self._finished_streaming_gens: List[ObjectRefGenerator] = []
        super().__init__(name, input_op, target_max_block_size)
        self._additional_split_factor = None

    def get_additional_split_factor(self) -> int:
        if self._additional_split_factor is None:
            return 1
        return self._additional_split_factor

    def set_additional_split_factor(self, k: int):
        self._additional_split_factor = k

    @property
    def name(self) -> str:
        name = super().name
        if self._additional_split_factor is not None:
            name += f'->SplitBlocks({self._additional_split_factor})'
        return name

    @classmethod
    def create(cls, map_transformer: MapTransformer, input_op: PhysicalOperator, target_max_block_size: Optional[int]=None, name: str='Map', compute_strategy: Optional[ComputeStrategy]=None, min_rows_per_bundle: Optional[int]=None, ray_remote_args: Optional[Dict[str, Any]]=None) -> 'MapOperator':
        """Create a MapOperator.

        This factory creates the MapOperator pool implementation that corresponds to the
        compute argument:
            - If None or TaskPoolStrategy -> TaskPoolMapOperator
            - If ActorPoolStrategy -> ActorPoolMapOperator

        Args:
            transform_fn: The function to apply to each ref bundle input.
            input_op: Operator generating input data for this op.
            init_fn: The callable class to instantiate if using ActorPoolMapOperator.
            name: The name of this operator.
            compute_strategy: Customize the compute strategy for this op.
            target_max_block_size: The target maximum number of bytes to
                include in an output block.
            min_rows_per_bundle: The number of rows to gather per batch passed to the
                transform_fn, or None to use the block size. Setting the batch size is
                important for the performance of GPU-accelerated transform functions.
                The actual rows passed may be less if the dataset is small.
            ray_remote_args: Customize the ray remote args for this op's tasks.
        """
        if compute_strategy is None:
            compute_strategy = TaskPoolStrategy()
        if isinstance(compute_strategy, TaskPoolStrategy):
            from ray.data._internal.execution.operators.task_pool_map_operator import TaskPoolMapOperator
            return TaskPoolMapOperator(map_transformer, input_op, name=name, target_max_block_size=target_max_block_size, min_rows_per_bundle=min_rows_per_bundle, ray_remote_args=ray_remote_args)
        elif isinstance(compute_strategy, ActorPoolStrategy):
            from ray.data._internal.execution.operators.actor_pool_map_operator import ActorPoolMapOperator, AutoscalingConfig, AutoscalingPolicy
            autoscaling_config = AutoscalingConfig.from_compute_strategy(compute_strategy)
            autoscaling_policy = AutoscalingPolicy(autoscaling_config)
            return ActorPoolMapOperator(map_transformer, input_op, autoscaling_policy=autoscaling_policy, name=name, target_max_block_size=target_max_block_size, min_rows_per_bundle=min_rows_per_bundle, ray_remote_args=ray_remote_args)
        else:
            raise ValueError(f'Unsupported execution strategy {compute_strategy}')

    def start(self, options: 'ExecutionOptions'):
        super().start(options)
        if options.preserve_order:
            self._output_queue = _OrderedOutputQueue()
        else:
            self._output_queue = _UnorderedOutputQueue()
        if options.locality_with_output:
            if isinstance(options.locality_with_output, list):
                locs = options.locality_with_output
            else:
                locs = [ray.get_runtime_context().get_node_id()]

            class RoundRobinAssign:

                def __init__(self, locs):
                    self.locs = locs
                    self.i = 0

                def __call__(self, args):
                    args = copy.deepcopy(args)
                    args['scheduling_strategy'] = NodeAffinitySchedulingStrategy(self.locs[self.i], soft=True, _spill_on_unavailable=True)
                    self.i += 1
                    self.i %= len(self.locs)
                    return args
            self._ray_remote_args_factory = RoundRobinAssign(locs)
        map_transformer = self._map_transformer
        if self.get_additional_split_factor() > 1:
            split_transformer = MapTransformer([ApplyAdditionalSplitToOutputBlocks(self.get_additional_split_factor())])
            map_transformer = map_transformer.fuse(split_transformer)
        self._map_transformer_ref = ray.put(map_transformer)

    def _add_input_inner(self, refs: RefBundle, input_index: int):
        assert input_index == 0, input_index
        self._block_ref_bundler.add_bundle(refs)
        if self._block_ref_bundler.has_bundle():
            bundle = self._block_ref_bundler.get_next_bundle()
            self._add_bundled_input(bundle)

    def _get_runtime_ray_remote_args(self, input_bundle: Optional[RefBundle]=None) -> Dict[str, Any]:
        ray_remote_args = copy.deepcopy(self._ray_remote_args)
        if 'scheduling_strategy' not in ray_remote_args:
            ctx = DataContext.get_current()
            if input_bundle and input_bundle.size_bytes() > ctx.large_args_threshold:
                ray_remote_args['scheduling_strategy'] = ctx.scheduling_strategy_large_args
                self._remote_args_for_metrics = copy.deepcopy(ray_remote_args)
            else:
                ray_remote_args['scheduling_strategy'] = ctx.scheduling_strategy
                if 'scheduling_strategy' not in self._remote_args_for_metrics:
                    self._remote_args_for_metrics = copy.deepcopy(ray_remote_args)
        if self._ray_remote_args_factory:
            return self._ray_remote_args_factory(ray_remote_args)
        return ray_remote_args

    @abstractmethod
    def _add_bundled_input(self, refs: RefBundle):
        """Add a pre-bundled upstream output to this operator.

        Unlike the add_input() arg, this RefBundle has already been further bundled by
        _block_ref_bundler up to the target size, meaning that this bundle is ready for
        task submission.

        This must be implemented by subclasses.

        Args:
            refs: The fully-bundled ref bundle that should be added as input.
        """
        raise NotImplementedError

    def _submit_data_task(self, gen: ObjectRefGenerator, inputs: RefBundle, task_done_callback: Optional[Callable[[], None]]=None):
        """Submit a new data-handling task."""
        task_index = self._next_data_task_idx
        self._next_data_task_idx += 1
        self._metrics.on_task_submitted(task_index, inputs)

        def _output_ready_callback(task_index, output: RefBundle):
            assert len(output) == 1
            self._metrics.on_output_generated(task_index, output)
            self._output_queue.notify_task_output_ready(task_index, output)

        def _task_done_callback(task_index: int, exception: Optional[Exception]):
            self._metrics.on_task_finished(task_index, exception)
            estimated_num_tasks = self.input_dependencies[0].num_outputs_total() / self._metrics.num_inputs_received * self._next_data_task_idx
            self._estimated_output_blocks = round(estimated_num_tasks * self._metrics.num_outputs_of_finished_tasks / self._metrics.num_tasks_finished)
            task = self._data_tasks.pop(task_index)
            self._finished_streaming_gens.append(task.get_waitable())
            self._output_queue.notify_task_completed(task_index)
            if task_done_callback:
                task_done_callback()
        self._data_tasks[task_index] = DataOpTask(task_index, gen, lambda output: _output_ready_callback(task_index, output), functools.partial(_task_done_callback, task_index))

    def _submit_metadata_task(self, result_ref: ObjectRef, task_done_callback: Callable[[], None]):
        """Submit a new metadata-handling task."""
        task_index = self._next_metadata_task_idx
        self._next_metadata_task_idx += 1

        def _task_done_callback():
            self._metadata_tasks.pop(task_index)
            task_done_callback()
        self._metadata_tasks[task_index] = MetadataOpTask(task_index, result_ref, _task_done_callback)

    def get_active_tasks(self) -> List[OpTask]:
        return list(self._metadata_tasks.values()) + list(self._data_tasks.values())

    def all_inputs_done(self):
        self._block_ref_bundler.done_adding_bundles()
        if self._block_ref_bundler.has_bundle():
            bundle = self._block_ref_bundler.get_next_bundle()
            self._add_bundled_input(bundle)
        super().all_inputs_done()

    def has_next(self) -> bool:
        assert self._started
        return self._output_queue.has_next()

    def _get_next_inner(self) -> RefBundle:
        assert self._started
        bundle = self._output_queue.get_next()
        for _, meta in bundle.blocks:
            self._output_metadata.append(meta)
        return bundle

    @abstractmethod
    def progress_str(self) -> str:
        raise NotImplementedError

    def _extra_metrics(self) -> Dict[str, Any]:
        return {'ray_remote_args': dict(sorted(self._remote_args_for_metrics.items()))}

    def get_stats(self) -> StatsDict:
        return {self._name: self._output_metadata}

    def get_map_transformer(self) -> MapTransformer:
        return self._map_transformer

    def shutdown(self):
        self._data_tasks.clear()
        self._metadata_tasks.clear()
        self._finished_streaming_gens.clear()

    @abstractmethod
    def current_resource_usage(self) -> ExecutionResources:
        raise NotImplementedError

    @abstractmethod
    def base_resource_usage(self) -> ExecutionResources:
        raise NotImplementedError

    @abstractmethod
    def incremental_resource_usage(self) -> ExecutionResources:
        raise NotImplementedError