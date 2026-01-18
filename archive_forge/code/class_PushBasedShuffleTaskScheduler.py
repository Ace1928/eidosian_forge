import logging
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union
import ray
from ray.data._internal.execution.interfaces import RefBundle, TaskContext
from ray.data._internal.planner.exchange.interfaces import (
from ray.data._internal.progress_bar import ProgressBar
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.stats import StatsDict
from ray.data.block import Block, BlockAccessor, BlockExecStats, BlockMetadata
from ray.data.context import DataContext
from ray.types import ObjectRef
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
class PushBasedShuffleTaskScheduler(ExchangeTaskScheduler):
    """
    Push-based shuffle merges intermediate map outputs on the reducer nodes
    while other map tasks are executing. The merged outputs are merged again
    during a final reduce stage. This works as follows:

    1. Submit rounds of concurrent map and merge tasks until all map inputs
    have been processed. In each round, we execute:

       M map tasks
         Each produces N outputs. Each output contains P blocks.
       N merge tasks
         Takes 1 output from each of M map tasks.
         Each produces P outputs.
       Where M and N are chosen to maximize parallelism across CPUs. Note that
       this assumes that all CPUs in the cluster will be dedicated to the
       shuffle job.

       Map and merge tasks are pipelined so that we always merge the previous
       round of map outputs while executing the next round of map tasks.

    2. In the final reduce stage:
       R reduce tasks
         Takes 1 output from one of the merge tasks from every round.

    Notes:
        N * P = R = total number of output blocks
        M / N = merge factor - the ratio of map : merge tasks is to improve
          pipelined parallelism. For example, if map takes twice as long to
          execute as merge, then we should set this to 2.
        See paper at https://arxiv.org/abs/2203.05072 for more details.
    """

    def execute(self, refs: List[RefBundle], output_num_blocks: int, ctx: TaskContext, map_ray_remote_args: Optional[Dict[str, Any]]=None, reduce_ray_remote_args: Optional[Dict[str, Any]]=None, merge_factor: int=2) -> Tuple[List[RefBundle], StatsDict]:
        logger.info('Using experimental push-based shuffle.')
        input_blocks_list = []
        for ref_bundle in refs:
            for block, _ in ref_bundle.blocks:
                input_blocks_list.append(block)
        input_owned = all((b.owns_blocks for b in refs))
        if map_ray_remote_args is None:
            map_ray_remote_args = {}
        if reduce_ray_remote_args is None:
            reduce_ray_remote_args = {}
        reduce_ray_remote_args = reduce_ray_remote_args.copy()
        reduce_ray_remote_args.pop('scheduling_strategy', None)
        num_cpus_per_node_map = _get_num_cpus_per_node_map()
        stage = self._compute_shuffle_schedule(num_cpus_per_node_map, len(input_blocks_list), merge_factor, output_num_blocks)
        map_fn = self._map_partition
        merge_fn = self._merge

        def map_partition(*args, **kwargs):
            return map_fn(self._exchange_spec.map, *args, **kwargs)

        def merge(*args, **kwargs):
            return merge_fn(self._exchange_spec.reduce, *args, **kwargs)
        shuffle_map = cached_remote_fn(map_partition)
        shuffle_map = shuffle_map.options(**map_ray_remote_args, num_returns=1 + stage.merge_schedule.num_merge_tasks_per_round)
        map_stage_iter = _MapStageIterator(input_blocks_list, shuffle_map, [output_num_blocks, stage.merge_schedule, *self._exchange_spec._map_args])
        sub_progress_bar_dict = ctx.sub_progress_bar_dict
        bar_name = ExchangeTaskSpec.MAP_SUB_PROGRESS_BAR_NAME
        assert bar_name in sub_progress_bar_dict, sub_progress_bar_dict
        map_bar = sub_progress_bar_dict[bar_name]
        map_stage_executor = _PipelinedStageExecutor(map_stage_iter, stage.num_map_tasks_per_round, progress_bar=map_bar)
        shuffle_merge = cached_remote_fn(merge)
        merge_stage_iter = _MergeStageIterator(map_stage_iter, shuffle_merge, stage, self._exchange_spec._reduce_args)
        merge_stage_executor = _PipelinedStageExecutor(merge_stage_iter, stage.merge_schedule.num_merge_tasks_per_round, max_concurrent_rounds=2)
        map_done = False
        merge_done = False
        map_stage_metadata = []
        merge_stage_metadata = []
        while not (map_done and merge_done):
            try:
                map_stage_metadata += next(map_stage_executor)
            except StopIteration:
                map_done = True
                break
            try:
                merge_stage_metadata += next(merge_stage_executor)
            except StopIteration:
                merge_done = True
                break
        all_merge_results = merge_stage_iter.pop_merge_results()
        bar_name = ExchangeTaskSpec.REDUCE_SUB_PROGRESS_BAR_NAME
        assert bar_name in sub_progress_bar_dict, sub_progress_bar_dict
        reduce_bar = sub_progress_bar_dict[bar_name]
        shuffle_reduce = cached_remote_fn(self._exchange_spec.reduce)
        reduce_stage_iter = _ReduceStageIterator(stage, shuffle_reduce, all_merge_results, reduce_ray_remote_args, self._exchange_spec._reduce_args)
        max_reduce_tasks_in_flight = output_num_blocks
        ctx = DataContext.get_current()
        if ctx.pipeline_push_based_shuffle_reduce_tasks:
            max_reduce_tasks_in_flight = min(max_reduce_tasks_in_flight, sum(num_cpus_per_node_map.values()))
        reduce_stage_executor = _PipelinedStageExecutor(reduce_stage_iter, max_reduce_tasks_in_flight, max_concurrent_rounds=2, progress_bar=reduce_bar)
        reduce_stage_metadata = []
        while True:
            try:
                reduce_stage_metadata += next(reduce_stage_executor)
            except StopIteration:
                break
        new_blocks = reduce_stage_iter.pop_reduce_results()
        sorted_blocks = [(block[0], block[1], reduce_stage_metadata[i]) for i, block in enumerate(new_blocks)]
        sorted_blocks.sort(key=lambda x: x[0])
        new_blocks, reduce_stage_metadata = ([], [])
        if sorted_blocks:
            _, new_blocks, reduce_stage_metadata = zip(*sorted_blocks)
        del sorted_blocks
        assert len(new_blocks) == output_num_blocks, f'Expected {output_num_blocks} outputs, produced {len(new_blocks)}'
        output = []
        for block, meta in zip(new_blocks, reduce_stage_metadata):
            output.append(RefBundle([(block, meta)], owns_blocks=input_owned))
        stats = {'map': map_stage_metadata, 'merge': merge_stage_metadata, 'reduce': reduce_stage_metadata}
        return (output, stats)

    @staticmethod
    def _map_partition(map_fn, idx: int, block: Block, output_num_blocks: int, schedule: _MergeTaskSchedule, *map_args: List[Any]) -> List[Union[BlockMetadata, Block]]:
        mapper_outputs = map_fn(idx, block, output_num_blocks, *map_args)
        meta = mapper_outputs.pop(-1)
        parts = []
        merge_idx = 0
        while mapper_outputs:
            partition_size = schedule.get_num_reducers_per_merge_idx(merge_idx)
            parts.append(mapper_outputs[:partition_size])
            mapper_outputs = mapper_outputs[partition_size:]
            merge_idx += 1
        assert len(parts) == schedule.num_merge_tasks_per_round, (len(parts), schedule.num_merge_tasks_per_round)
        return parts + [meta]

    @staticmethod
    def _merge(reduce_fn, *all_mapper_outputs: List[List[Block]], reduce_args: Optional[List[Any]]=None) -> List[Union[BlockMetadata, Block]]:
        """
        Returns list of [BlockMetadata, O1, O2, O3, ...output_num_blocks].
        """
        assert len({len(mapper_outputs) for mapper_outputs in all_mapper_outputs}) == 1, 'Received different number of map inputs'
        stats = BlockExecStats.builder()
        if not reduce_args:
            reduce_args = []
        num_rows = 0
        size_bytes = 0
        schema = None
        for i, mapper_outputs in enumerate(zip(*all_mapper_outputs)):
            block, meta = reduce_fn(*reduce_args, *mapper_outputs, partial_reduce=True)
            yield block
            block = BlockAccessor.for_block(block)
            num_rows += block.num_rows()
            size_bytes += block.size_bytes()
            schema = block.schema()
            del block
        yield BlockMetadata(num_rows=num_rows, size_bytes=size_bytes, schema=schema, input_files=None, exec_stats=stats.build())

    @staticmethod
    def _compute_shuffle_schedule(num_cpus_per_node_map: Dict[str, int], num_input_blocks: int, merge_factor: int, num_output_blocks: int) -> _PushBasedShuffleStage:
        num_cpus_total = sum((v for v in num_cpus_per_node_map.values()))
        task_parallelism = min(num_cpus_total, num_input_blocks)
        num_tasks_per_map_merge_group = merge_factor + 1
        num_merge_tasks_per_round = 0
        merge_task_placement = []
        leftover_cpus = 0
        for node, num_cpus in num_cpus_per_node_map.items():
            node_parallelism = min(num_cpus, num_input_blocks // len(num_cpus_per_node_map))
            num_merge_tasks = node_parallelism // num_tasks_per_map_merge_group
            for i in range(num_merge_tasks):
                merge_task_placement.append(node)
            num_merge_tasks_per_round += num_merge_tasks
            leftover_cpus += node_parallelism % num_tasks_per_map_merge_group
            if num_merge_tasks == 0 and leftover_cpus > num_tasks_per_map_merge_group:
                merge_task_placement.append(node)
                num_merge_tasks_per_round += 1
                leftover_cpus -= num_tasks_per_map_merge_group
        if num_merge_tasks_per_round == 0:
            merge_task_placement.append(list(num_cpus_per_node_map)[0])
            num_merge_tasks_per_round = 1
        assert num_merge_tasks_per_round == len(merge_task_placement)
        num_map_tasks_per_round = max(task_parallelism - num_merge_tasks_per_round, 1)
        num_rounds = math.ceil(num_input_blocks / num_map_tasks_per_round)
        return _PushBasedShuffleStage(num_output_blocks, num_rounds, num_map_tasks_per_round, merge_task_placement)