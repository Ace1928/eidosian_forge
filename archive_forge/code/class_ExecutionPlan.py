import copy
import functools
import itertools
from typing import (
import ray
from ray._private.internal_api import get_memory_info_reply, get_state_from_address
from ray.data._internal.block_list import BlockList
from ray.data._internal.compute import (
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.execution.interfaces import TaskContext
from ray.data._internal.lazy_block_list import LazyBlockList
from ray.data._internal.logical.operators.input_data_operator import InputData
from ray.data._internal.logical.operators.read_operator import Read
from ray.data._internal.logical.rules.operator_fusion import _are_remote_args_compatible
from ray.data._internal.logical.rules.set_read_parallelism import (
from ray.data._internal.planner.plan_read_op import (
from ray.data._internal.stats import DatasetStats, DatasetStatsSummary
from ray.data._internal.util import (
from ray.data.block import Block, BlockMetadata
from ray.data.context import DataContext
from ray.types import ObjectRef
from ray.util.debug import log_once
class ExecutionPlan:
    """A lazy execution plan for a Dataset."""

    def __init__(self, in_blocks: BlockList, stats: DatasetStats, *, run_by_consumer: bool, data_context: Optional[DataContext]=None):
        """Create a plan with no transformation stages.

        Args:
            in_blocks: Base list of blocks.
            stats: Stats for the base blocks.
            dataset_uuid: Dataset's UUID.
            run_by_consumer: Whether this plan is invoked to run by the consumption
            APIs (e.g. .iter_batches()).
        """
        self._in_blocks = in_blocks
        self._in_stats = stats
        self._snapshot_blocks = None
        self._snapshot_stats = None
        self._stages_before_snapshot = []
        self._stages_after_snapshot = []
        self._last_optimized_stages = None
        self._schema = None
        self._dataset_uuid = None
        self._run_by_consumer = run_by_consumer
        self._dataset_name = None
        if data_context is None:
            self._context = copy.deepcopy(DataContext.get_current())
        else:
            self._context = data_context

    def __repr__(self) -> str:
        return f'ExecutionPlan(dataset_uuid={self._dataset_uuid}, run_by_consumer={self._run_by_consumer}, in_blocks={self._in_blocks}, stages_before_snapshot={self._stages_before_snapshot}, stages_after_snapshot={self._stages_after_snapshot}, snapshot_blocks={self._snapshot_blocks})'

    def get_plan_as_string(self, classname: str) -> str:
        """Create a cosmetic string representation of this execution plan.

        Returns:
            The string representation of this execution plan.
        """
        plan_str = ''
        num_stages = 0
        dataset_blocks = None
        if self._stages_after_snapshot:
            for stage in self._stages_after_snapshot[::-1]:
                stage_str = stage.name.split('(')
                stage_str[0] = capitalize(stage_str[0])
                stage_name = '('.join(stage_str)
                if num_stages == 0:
                    plan_str += f'{stage_name}\n'
                else:
                    trailing_space = ' ' * ((num_stages - 1) * 3)
                    plan_str += f'{trailing_space}+- {stage_name}\n'
                num_stages += 1
            if self._snapshot_blocks is not None:
                schema = self._get_unified_blocks_schema(self._snapshot_blocks, fetch_if_missing=False)
                dataset_blocks = self._snapshot_blocks
            else:
                assert self._in_blocks is not None
                schema = self._get_unified_blocks_schema(self._in_blocks, fetch_if_missing=False)
                dataset_blocks = self._in_blocks
        else:
            schema = self.schema(fetch_if_missing=False)
            dataset_blocks = self._snapshot_blocks
        if schema is None:
            schema_str = 'Unknown schema'
        elif isinstance(schema, type):
            schema_str = str(schema)
        else:
            schema_str = []
            for n, t in zip(schema.names, schema.types):
                if hasattr(t, '__name__'):
                    t = t.__name__
                schema_str.append(f'{n}: {t}')
            schema_str = ', '.join(schema_str)
            schema_str = '{' + schema_str + '}'
        count = self._get_num_rows_from_blocks_metadata(dataset_blocks)
        if count is None:
            count = '?'
        if dataset_blocks is None:
            num_blocks = '?'
        else:
            num_blocks = dataset_blocks.estimated_num_blocks()
        name_str = 'name={}, '.format(self._dataset_name) if self._dataset_name is not None else ''
        dataset_str = '{}({}num_blocks={}, num_rows={}, schema={})'.format(classname, name_str, num_blocks, count, schema_str)
        SCHEMA_LINE_CHAR_LIMIT = 80
        MIN_FIELD_LENGTH = 10
        INDENT_STR = ' ' * 3
        trailing_space = ' ' * (max(num_stages, 0) * 3)
        if len(dataset_str) > SCHEMA_LINE_CHAR_LIMIT:
            schema_str_on_new_line = f'{trailing_space}{INDENT_STR}schema={schema_str}'
            if len(schema_str_on_new_line) > SCHEMA_LINE_CHAR_LIMIT:
                schema_str = []
                for n, t in zip(schema.names, schema.types):
                    if hasattr(t, '__name__'):
                        t = t.__name__
                    col_str = f'{trailing_space}{INDENT_STR * 2}{n}: {t}'
                    if len(col_str) > SCHEMA_LINE_CHAR_LIMIT:
                        shortened_suffix = f'...: {str(t)}'
                        chars_left_for_col_name = max(SCHEMA_LINE_CHAR_LIMIT - len(shortened_suffix), MIN_FIELD_LENGTH)
                        col_str = f'{col_str[:chars_left_for_col_name]}{shortened_suffix}'
                    schema_str.append(col_str)
                schema_str = ',\n'.join(schema_str)
                schema_str = '{\n' + schema_str + f'\n{trailing_space}{INDENT_STR}' + '}'
            name_str = f'\n{trailing_space}{INDENT_STR}name={self._dataset_name},' if self._dataset_name is not None else ''
            dataset_str = f'{classname}({name_str}\n{trailing_space}{INDENT_STR}num_blocks={num_blocks},\n{trailing_space}{INDENT_STR}num_rows={count},\n{trailing_space}{INDENT_STR}schema={schema_str}\n{trailing_space})'
        if num_stages == 0:
            plan_str = dataset_str
        else:
            trailing_space = ' ' * ((num_stages - 1) * 3)
            plan_str += f'{trailing_space}+- {dataset_str}'
        return plan_str

    def with_stage(self, stage: 'Stage') -> 'ExecutionPlan':
        """Return a copy of this plan with the given stage appended.

        Args:
            stage: The stage to append.

        Returns:
            A new ExecutionPlan with this stage appended.
        """
        copy = self.copy()
        copy._stages_after_snapshot.append(stage)
        return copy

    def link_logical_plan(self, logical_plan):
        """Link the logical plan into this execution plan.

        This is used for triggering execution for optimizer code path in this legacy
        execution plan.
        """
        self._logical_plan = logical_plan

    def copy(self) -> 'ExecutionPlan':
        """Create a shallow copy of this execution plan.

        This copy can be executed without mutating the original, but clearing the copy
        will also clear the original.

        Returns:
            A shallow copy of this execution plan.
        """
        plan_copy = ExecutionPlan(self._in_blocks, self._in_stats, run_by_consumer=self._run_by_consumer, data_context=self._context)
        if self._snapshot_blocks is not None:
            plan_copy._snapshot_blocks = self._snapshot_blocks
            plan_copy._snapshot_stats = self._snapshot_stats
        plan_copy._stages_before_snapshot = self._stages_before_snapshot.copy()
        plan_copy._stages_after_snapshot = self._stages_after_snapshot.copy()
        plan_copy._dataset_name = self._dataset_name
        return plan_copy

    def deep_copy(self) -> 'ExecutionPlan':
        """Create a deep copy of this execution plan.

        This copy can be executed AND cleared without mutating the original.

        Returns:
            A deep copy of this execution plan.
        """
        in_blocks = self._in_blocks
        if isinstance(in_blocks, BlockList):
            in_blocks = in_blocks.copy()
        plan_copy = ExecutionPlan(in_blocks, copy.copy(self._in_stats), run_by_consumer=self._run_by_consumer)
        if self._snapshot_blocks:
            plan_copy._snapshot_blocks = self._snapshot_blocks.copy()
            plan_copy._snapshot_stats = copy.copy(self._snapshot_stats)
        plan_copy._stages_before_snapshot = self._stages_before_snapshot.copy()
        plan_copy._stages_after_snapshot = self._stages_after_snapshot.copy()
        plan_copy._dataset_name = self._dataset_name
        return plan_copy

    def initial_num_blocks(self) -> int:
        """Get the estimated number of blocks after applying all plan stages."""
        if self.has_computed_output():
            return self._snapshot_blocks.initial_num_blocks()
        for stage in self._stages_after_snapshot[::-1]:
            if stage.num_blocks is not None:
                return stage.num_blocks
        if self._snapshot_blocks is not None:
            return self._snapshot_blocks.initial_num_blocks()
        for stage in self._stages_before_snapshot[::-1]:
            if stage.num_blocks is not None:
                return stage.num_blocks
        if self._in_blocks is not None:
            return self._in_blocks.initial_num_blocks()
        return None

    def schema(self, fetch_if_missing: bool=False) -> Union[type, 'pyarrow.lib.Schema']:
        """Get the schema after applying all plan stages.

        Args:
            fetch_if_missing: Whether to execute the plan to fetch the schema.

        Returns:
            The schema of the output dataset.
        """
        from ray.data._internal.stage_impl import RandomizeBlocksStage
        if self._schema is not None:
            return self._schema
        if self._stages_after_snapshot:
            if fetch_if_missing:
                if isinstance(self._stages_after_snapshot[-1], RandomizeBlocksStage):
                    a = self._stages_after_snapshot.pop()
                    try:
                        self.execute()
                    finally:
                        self._stages_after_snapshot.append(a)
                else:
                    self.execute()
            elif len(self._stages_after_snapshot) == 1 and isinstance(self._stages_after_snapshot[-1], RandomizeBlocksStage):
                self.execute()
            else:
                return None
        elif self._in_blocks is not None and self._snapshot_blocks is None:
            self.execute()
        blocks = self._snapshot_blocks
        if not blocks:
            return None
        self._schema = self._get_unified_blocks_schema(blocks, fetch_if_missing)
        return self._schema

    def cache_schema(self, schema: Union[type, 'pyarrow.lib.Schema']):
        self._schema = schema

    def _get_unified_blocks_schema(self, blocks: BlockList, fetch_if_missing: bool=False) -> Union[type, 'pyarrow.lib.Schema']:
        """Get the unified schema of the blocks.

        Args:
            blocks: the blocks to get schema
            fetch_if_missing: Whether to execute the blocks to fetch the schema.
        """
        if isinstance(blocks, LazyBlockList):
            blocks.ensure_metadata_for_first_block()
        metadata = blocks.get_metadata(fetch_if_missing=False)
        unified_schema = unify_block_metadata_schema(metadata)
        if unified_schema is not None:
            return unified_schema
        if not fetch_if_missing:
            return None
        for _, m in blocks.iter_blocks_with_metadata():
            if m.schema is not None and (m.num_rows is None or m.num_rows > 0):
                return m.schema
        return None

    def meta_count(self) -> Optional[int]:
        """Get the number of rows after applying all plan stages if possible.

        This method will never trigger any computation.

        Returns:
            The number of records of the result Dataset, or None.
        """
        if self._stages_after_snapshot:
            return None
        elif self._in_blocks is not None and self._snapshot_blocks is None:
            self.execute()
        return self._get_num_rows_from_blocks_metadata(self._snapshot_blocks)

    def _get_num_rows_from_blocks_metadata(self, blocks: BlockList) -> Optional[int]:
        metadata = blocks.get_metadata() if blocks else None
        if metadata and all((m.num_rows is not None for m in metadata)):
            return sum((m.num_rows for m in metadata))
        else:
            return None

    def execute_to_iterator(self, allow_clear_input_blocks: bool=True, force_read: bool=False) -> Tuple[Iterator[Tuple[ObjectRef[Block], BlockMetadata]], DatasetStats, Optional['Executor']]:
        """Execute this plan, returning an iterator.

        If the streaming execution backend is enabled, this will use streaming
        execution to generate outputs, otherwise it will fall back to bulk exec.

        Args:
            allow_clear_input_blocks: Whether we should try to clear the input blocks
                for each stage.
            force_read: Whether to force the read stage to fully execute.

        Returns:
            Tuple of iterator over output blocks and the executor.
        """
        ctx = self._context
        if not ctx.use_streaming_executor or self.has_computed_output():
            return (self.execute(allow_clear_input_blocks, force_read).iter_blocks_with_metadata(), self._snapshot_stats, None)
        from ray.data._internal.execution.legacy_compat import execute_to_legacy_block_iterator
        from ray.data._internal.execution.streaming_executor import StreamingExecutor
        metrics_tag = create_dataset_tag(self._dataset_name, self._dataset_uuid)
        executor = StreamingExecutor(copy.deepcopy(ctx.execution_options), metrics_tag)
        block_iter = execute_to_legacy_block_iterator(executor, self, allow_clear_input_blocks=allow_clear_input_blocks, dataset_uuid=self._dataset_uuid)
        gen = iter(block_iter)
        try:
            block_iter = itertools.chain([next(gen)], gen)
        except StopIteration:
            pass
        self._snapshot_stats = executor.get_stats()
        return (block_iter, self._snapshot_stats, executor)

    def execute(self, allow_clear_input_blocks: bool=True, force_read: bool=False, preserve_order: bool=False) -> BlockList:
        """Execute this plan.

        Args:
            allow_clear_input_blocks: Whether we should try to clear the input blocks
                for each stage.
            force_read: Whether to force the read stage to fully execute.
            preserve_order: Whether to preserve order in execution.

        Returns:
            The blocks of the output dataset.
        """
        context = self._context
        if not ray.available_resources().get('CPU'):
            if log_once('cpu_warning'):
                logger.get_logger().warning('Warning: The Ray cluster currently does not have any available CPUs. The Dataset job will hang unless more CPUs are freed up. A common reason is that cluster resources are used by Actors or Tune trials; see the following link for more details: https://docs.ray.io/en/latest/data/data-internals.html#ray-data-and-tune')
        if not self.has_computed_output():
            if self._run_with_new_execution_backend():
                from ray.data._internal.execution.legacy_compat import _get_initial_stats_from_plan, execute_to_legacy_block_list, get_legacy_lazy_block_list_read_only
                if self._is_input_data_only():
                    blocks = self._in_blocks
                    stats = _get_initial_stats_from_plan(self)
                elif self.is_read_only():
                    blocks = get_legacy_lazy_block_list_read_only(self)
                    stats = _get_initial_stats_from_plan(self)
                else:
                    from ray.data._internal.execution.streaming_executor import StreamingExecutor
                    metrics_tag = create_dataset_tag(self._dataset_name, self._dataset_uuid)
                    executor = StreamingExecutor(copy.deepcopy(context.execution_options), metrics_tag)
                    blocks = execute_to_legacy_block_list(executor, self, allow_clear_input_blocks=allow_clear_input_blocks, dataset_uuid=self._dataset_uuid, preserve_order=preserve_order)
                    stats = executor.get_stats()
                    stats_summary_string = stats.to_summary().to_string(include_parent=False)
                    logger.get_logger(log_to_stdout=context.enable_auto_log_stats).info(stats_summary_string)
                if not self._run_by_consumer:
                    blocks._owned_by_consumer = False
            else:
                blocks, stats, stages = self._optimize()
                for stage_idx, stage in enumerate(stages):
                    if allow_clear_input_blocks:
                        clear_input_blocks = self._should_clear_input_blocks(blocks, stage_idx)
                    else:
                        clear_input_blocks = False
                    stats_builder = stats.child_builder(stage.name)
                    blocks, stage_info = stage(blocks, clear_input_blocks, self._run_by_consumer)
                    if stage_info:
                        stats = stats_builder.build_multistage(stage_info)
                    else:
                        stats = stats_builder.build(blocks)
                    stats.dataset_uuid = self._dataset_uuid
                    stats_summary_string = stats.to_summary().to_string(include_parent=False)
                    logger.get_logger(log_to_stdout=context.enable_auto_log_stats).info(stats_summary_string)
            try:
                reply = get_memory_info_reply(get_state_from_address(ray.get_runtime_context().gcs_address))
                if reply.store_stats.spill_time_total_s > 0:
                    stats.global_bytes_spilled = int(reply.store_stats.spilled_bytes_total)
                if reply.store_stats.restore_time_total_s > 0:
                    stats.global_bytes_restored = int(reply.store_stats.restored_bytes_total)
            except Exception as e:
                logger.get_logger(log_to_stdout=False).warning(f'Skipping recording memory spilled and restored statistics due to exception: {e}')
            stats.dataset_bytes_spilled = 0

            def collect_stats(cur_stats):
                stats.dataset_bytes_spilled += cur_stats.extra_metrics.get('obj_store_mem_spilled', 0)
                for parent in cur_stats.parents:
                    collect_stats(parent)
            collect_stats(stats)
            self._snapshot_blocks = blocks
            self._snapshot_stats = stats
            self._snapshot_stats.dataset_uuid = self._dataset_uuid
            self._stages_before_snapshot += self._stages_after_snapshot
            self._stages_after_snapshot = []
            if self.is_read_only():
                self._in_blocks = blocks
        if _is_lazy(self._snapshot_blocks) and force_read:
            executed_blocks = self._snapshot_blocks.compute_to_blocklist()
            self._snapshot_stats = self._snapshot_blocks.stats()
            self._snapshot_blocks = executed_blocks
            if self.is_read_only():
                self._in_blocks = self._snapshot_blocks
        return self._snapshot_blocks

    def clear_block_refs(self) -> None:
        """Clear all cached block references of this plan, including input blocks.

        This will render the plan un-executable unless the root is a LazyBlockList."""
        self._in_blocks.clear()
        self._clear_snapshot()

    def _clear_snapshot(self) -> None:
        """Clear the snapshot kept in the plan to the beginning state."""
        self._snapshot_blocks = None
        self._snapshot_stats = None
        self._stages_after_snapshot = self._stages_before_snapshot + self._stages_after_snapshot
        self._stages_before_snapshot = []

    def stats(self) -> DatasetStats:
        """Return stats for this plan.

        If the plan isn't executed, an empty stats object will be returned.
        """
        if not self._snapshot_stats:
            return DatasetStats(stages={}, parent=None)
        return self._snapshot_stats

    def stats_summary(self) -> DatasetStatsSummary:
        return self.stats().to_summary()

    def _should_clear_input_blocks(self, blocks: BlockList, stage_idx: int):
        """Whether the provided blocks should be cleared when passed into the stage.

        Args:
            blocks: The blocks that we may want to clear.
            stage_idx: The position of the stage in the optimized after-snapshot chain.
        """
        if stage_idx != 0 or self._stages_before_snapshot:
            return True
        elif isinstance(blocks, LazyBlockList):
            return True
        else:
            return False

    def _optimize(self) -> Tuple[BlockList, DatasetStats, List[Stage]]:
        """Apply stage fusion optimizations, returning an updated source block list and
        associated stats, and a set of optimized stages.
        """
        context = self._context
        blocks, stats, stages = self._get_source_blocks_and_stages()
        logical_op = self._logical_plan.dag
        if isinstance(logical_op, Read) and isinstance(blocks, LazyBlockList):
            ctx = DataContext.get_current()
            detected_parallelism, reason, estimated_num_blocks, k = compute_additional_split_factor(logical_op._datasource_or_legacy_reader, logical_op._parallelism, logical_op._mem_size, ctx.target_max_block_size, cur_additional_split_factor=None)
            if logical_op._parallelism == -1:
                assert reason != ''
                logger.get_logger().info(f'Using autodetected parallelism={detected_parallelism} for stage {logical_op.name} to satisfy {reason}.')
            if k is not None:
                logger.get_logger().info(f'To satisfy the requested parallelism of {detected_parallelism}, each read task output is split into {k} smaller blocks.')
            for read_task in blocks._tasks:
                apply_output_blocks_handling_to_read_task(read_task, k)
            blocks._estimated_num_blocks = estimated_num_blocks
        if context.optimize_reorder_stages:
            stages = _reorder_stages(stages)
        if context.optimize_fuse_stages:
            if context.optimize_fuse_read_stages:
                blocks, stats, stages = _rewrite_read_stages(blocks, stats, stages, self._dataset_uuid)
            stages = _fuse_one_to_one_stages(stages)
            self._last_optimized_stages = stages
        return (blocks, stats, stages)

    def _get_source_blocks_and_stages(self) -> Tuple[BlockList, DatasetStats, List[Stage]]:
        """Get the source blocks, corresponding stats, and the stages for plan
        execution.

        If a computed snapshot exists and has not been cleared, return the snapshot
        blocks and stats; otherwise, return the input blocks and stats that the plan was
        created with.
        """
        stages = self._stages_after_snapshot.copy()
        if self._snapshot_blocks is not None:
            if not self._snapshot_blocks.is_cleared():
                blocks = self._snapshot_blocks
                stats = self._snapshot_stats
                self._clear_snapshot()
            else:
                blocks = self._in_blocks
                stats = self._in_stats
                stages = self._stages_before_snapshot + self._stages_after_snapshot
        else:
            blocks = self._in_blocks
            stats = self._in_stats
        return (blocks, stats, stages)

    def has_lazy_input(self) -> bool:
        """Return whether this plan has lazy input blocks."""
        return _is_lazy(self._in_blocks)

    def is_read_only(self) -> bool:
        """Return whether the underlying logical plan contains only a Read op."""
        root_op = self._logical_plan.dag
        return isinstance(root_op, Read) and len(root_op.input_dependencies) == 0

    def _is_input_data_only(self) -> bool:
        """Return whether the underlying logical plan contains only an InputData op
        (e.g. in the case of a :class:`~ray.data.MaterializedDataset`)."""
        root_op = self._logical_plan.dag
        return isinstance(root_op, InputData) and len(root_op.input_dependencies) == 0

    def is_read_stage_equivalent(self) -> bool:
        """Return whether this plan can be executed as only a read stage."""
        from ray.data._internal.stage_impl import RandomizeBlocksStage
        context = self._context
        remaining_stages = self._stages_after_snapshot
        if context.optimize_fuse_stages and remaining_stages and isinstance(remaining_stages[0], RandomizeBlocksStage):
            remaining_stages = remaining_stages[1:]
        return self.has_lazy_input() and (not self._stages_before_snapshot) and (not remaining_stages) and (not self._snapshot_blocks or isinstance(self._snapshot_blocks, LazyBlockList))

    def has_computed_output(self) -> bool:
        """Whether this plan has a computed snapshot for the final stage, i.e. for the
        output of this plan.
        """
        return self._snapshot_blocks is not None and (not self._stages_after_snapshot) and (not self._snapshot_blocks.is_cleared())

    def _run_with_new_execution_backend(self) -> bool:
        """Whether this plan should run with new backend.
        By default, the new execution backend is now fully enabled
        unless configured otherwise by the user."""
        return self._context.new_execution_backend

    def require_preserve_order(self) -> bool:
        """Whether this plan requires to preserve order when running with new
        backend.
        """
        from ray.data._internal.stage_impl import SortStage, ZipStage
        for stage in self._stages_after_snapshot:
            if isinstance(stage, ZipStage) or isinstance(stage, SortStage):
                return True
        return False