import os
import threading
import time
import uuid
from typing import Dict, Iterator, List, Optional
import ray
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.execution.autoscaling_requester import (
from ray.data._internal.execution.backpressure_policy import (
from ray.data._internal.execution.interfaces import (
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.execution.streaming_executor_state import (
from ray.data._internal.progress_bar import ProgressBar
from ray.data._internal.stats import DatasetStats, StatsManager
from ray.data.context import DataContext
class StreamingExecutor(Executor, threading.Thread):
    """A streaming Dataset executor.

    This implementation executes Dataset DAGs in a fully streamed way. It runs
    by setting up the operator topology, and then routing blocks through operators in
    a way that maximizes throughput under resource constraints.
    """

    def __init__(self, options: ExecutionOptions, dataset_tag: str='unknown_dataset'):
        self._start_time: Optional[float] = None
        self._initial_stats: Optional[DatasetStats] = None
        self._final_stats: Optional[DatasetStats] = None
        self._global_info: Optional[ProgressBar] = None
        self._execution_id = uuid.uuid4().hex
        self._autoscaling_state = AutoscalingState()
        self._shutdown_lock = threading.RLock()
        self._execution_started = False
        self._shutdown = False
        self._topology: Optional[Topology] = None
        self._output_node: Optional[OpState] = None
        self._backpressure_policies: List[BackpressurePolicy] = []
        self._dataset_tag = dataset_tag
        self._has_op_completed: Optional[Dict[PhysicalOperator, bool]] = None
        self._max_errored_blocks = DataContext.get_current().max_errored_blocks
        self._num_errored_blocks = 0
        self._last_debug_log_time = 0
        Executor.__init__(self, options)
        thread_name = f'StreamingExecutor-{self._execution_id}'
        threading.Thread.__init__(self, daemon=True, name=thread_name)

    def execute(self, dag: PhysicalOperator, initial_stats: Optional[DatasetStats]=None) -> Iterator[RefBundle]:
        """Executes the DAG using a streaming execution strategy.

        We take an event-loop approach to scheduling. We block on the next scheduling
        event using `ray.wait`, updating operator state and dispatching new tasks.
        """
        self._initial_stats = initial_stats
        self._start_time = time.perf_counter()
        if not isinstance(dag, InputDataBuffer):
            logger.get_logger().info('Executing DAG %s', dag)
            logger.get_logger().info('Execution config: %s', self._options)
            if not self._options.verbose_progress:
                logger.get_logger().info('Tip: For detailed progress reporting, run `ray.data.DataContext.get_current().execution_options.verbose_progress = True`')
        self._topology, _ = build_streaming_topology(dag, self._options)
        self._backpressure_policies = get_backpressure_policies(self._topology)
        self._has_op_completed = {op: False for op in self._topology}
        if not isinstance(dag, InputDataBuffer):
            self._global_info = ProgressBar('Running', dag.num_outputs_total())
        self._output_node: OpState = self._topology[dag]
        StatsManager.register_dataset_to_stats_actor(self._dataset_tag, self._get_operator_tags())
        self.start()
        self._execution_started = True

        class StreamIterator(OutputIterator):

            def __init__(self, outer: Executor):
                self._outer = outer

            def get_next(self, output_split_idx: Optional[int]=None) -> RefBundle:
                try:
                    item = self._outer._output_node.get_output_blocking(output_split_idx)
                    if self._outer._global_info:
                        self._outer._global_info.update(1, dag._estimated_output_blocks)
                    return item
                except BaseException as e:
                    self._outer.shutdown(isinstance(e, StopIteration))
                    raise

            def __del__(self):
                self._outer.shutdown()
        return StreamIterator(self)

    def __del__(self):
        self.shutdown()

    def shutdown(self, execution_completed: bool=True):
        context = DataContext.get_current()
        global _num_shutdown
        with self._shutdown_lock:
            if not self._execution_started or self._shutdown:
                return
            logger.get_logger().debug(f'Shutting down {self}.')
            _num_shutdown += 1
            self._shutdown = True
            self.join(timeout=2.0)
            self._update_stats_metrics(state='FINISHED' if execution_completed else 'FAILED', force_update=True)
            StatsManager.clear_execution_metrics(self._dataset_tag, self._get_operator_tags())
            self._final_stats = self._generate_stats()
            stats_summary_string = self._final_stats.to_summary().to_string(include_parent=False)
            logger.get_logger(log_to_stdout=context.enable_auto_log_stats).info(stats_summary_string)
            if self._global_info:
                self._global_info.close()
            for op, state in self._topology.items():
                op.shutdown()
                state.close_progress_bars()
            actor = get_or_create_autoscaling_requester_actor()
            actor.request_resources.remote({}, self._execution_id)

    def run(self):
        """Run the control loop in a helper thread.

        Results are returned via the output node's outqueue.
        """
        try:
            while self._scheduling_loop_step(self._topology) and (not self._shutdown):
                pass
        except Exception as e:
            self._output_node.mark_finished(e)
        finally:
            self._output_node.mark_finished()

    def get_stats(self):
        """Return the stats object for the streaming execution.

        The stats object will be updated as streaming execution progresses.
        """
        if self._final_stats:
            return self._final_stats
        else:
            return self._generate_stats()

    def _generate_stats(self) -> DatasetStats:
        """Create a new stats object reflecting execution status so far."""
        stats = self._initial_stats or DatasetStats(stages={}, parent=None)
        for op in self._topology:
            if isinstance(op, InputDataBuffer):
                continue
            builder = stats.child_builder(op.name, override_start_time=self._start_time)
            stats = builder.build_multistage(op.get_stats())
            stats.extra_metrics = op.metrics.as_dict()
        return stats

    def _scheduling_loop_step(self, topology: Topology) -> bool:
        """Run one step of the scheduling loop.

        This runs a few general phases:
            1. Waiting for the next task completion using `ray.wait()`.
            2. Pulling completed refs into operator outqueues.
            3. Selecting and dispatching new inputs to operators.

        Returns:
            True if we should continue running the scheduling loop.
        """
        if DEBUG_TRACE_SCHEDULING:
            logger.get_logger().info('Scheduling loop step...')
        num_errored_blocks = process_completed_tasks(topology, self._backpressure_policies, self._max_errored_blocks)
        if self._max_errored_blocks > 0:
            self._max_errored_blocks -= num_errored_blocks
        self._num_errored_blocks += num_errored_blocks
        limits = self._get_or_refresh_resource_limits()
        cur_usage = TopologyResourceUsage.of(topology)
        self._report_current_usage(cur_usage, limits)
        op = select_operator_to_run(topology, cur_usage, limits, self._backpressure_policies, ensure_at_least_one_running=self._consumer_idling(), execution_id=self._execution_id, autoscaling_state=self._autoscaling_state)
        i = 0
        while op is not None:
            i += 1
            if i > PROGRESS_BAR_UPDATE_INTERVAL:
                break
            if DEBUG_TRACE_SCHEDULING:
                _debug_dump_topology(topology)
            topology[op].dispatch_next_task()
            cur_usage = TopologyResourceUsage.of(topology)
            op = select_operator_to_run(topology, cur_usage, limits, self._backpressure_policies, ensure_at_least_one_running=self._consumer_idling(), execution_id=self._execution_id, autoscaling_state=self._autoscaling_state)
        update_operator_states(topology)
        for op_state in topology.values():
            op_state.refresh_progress_bar()
        self._update_stats_metrics(state='RUNNING')
        if time.time() - self._last_debug_log_time >= DEBUG_LOG_INTERVAL_SECONDS:
            _log_op_metrics(topology)
            if not DEBUG_TRACE_SCHEDULING:
                _debug_dump_topology(topology, log_to_stdout=False)
            self._last_debug_log_time = time.time()
        for op in topology:
            if op.completed() and (not self._has_op_completed[op]):
                log_str = f'Operator {op} completed. Operator Metrics:\n{op._metrics.as_dict()}'
                logger.get_logger(log_to_stdout=False).info(log_str)
                self._has_op_completed[op] = True
        return not all((op.completed() for op in topology))

    def _consumer_idling(self) -> bool:
        """Returns whether the user thread is blocked on topology execution."""
        return len(self._output_node.outqueue) == 0

    def _get_or_refresh_resource_limits(self) -> ExecutionResources:
        """Return concrete limits for use at the current time.

        This method autodetects any unspecified execution resource limits based on the
        current cluster size, refreshing these values periodically to support cluster
        autoscaling.
        """
        base = self._options.resource_limits
        exclude = self._options.exclude_resources
        cluster = ray.cluster_resources()
        cpu = base.cpu
        if cpu is None:
            cpu = cluster.get('CPU', 0.0) - (exclude.cpu or 0.0)
        gpu = base.gpu
        if gpu is None:
            gpu = cluster.get('GPU', 0.0) - (exclude.gpu or 0.0)
        object_store_memory = base.object_store_memory
        if object_store_memory is None:
            object_store_memory = round(DEFAULT_OBJECT_STORE_MEMORY_LIMIT_FRACTION * cluster.get('object_store_memory', 0.0)) - (exclude.object_store_memory or 0)
        return ExecutionResources(cpu=cpu, gpu=gpu, object_store_memory=object_store_memory)

    def _report_current_usage(self, cur_usage: TopologyResourceUsage, limits: ExecutionResources) -> None:
        resources_status = f'Running: {cur_usage.overall.cpu}/{limits.cpu} CPU, {cur_usage.overall.gpu}/{limits.gpu} GPU, {cur_usage.overall.object_store_memory_str()}/{limits.object_store_memory_str()} object_store_memory'
        if self._global_info:
            self._global_info.set_description(resources_status)

    def _get_operator_tags(self):
        """Returns a list of operator tags."""
        return [f'{op.name}{i}' for i, op in enumerate(self._topology)]

    def _get_state_dict(self, state):
        last_op, last_state = list(self._topology.items())[-1]
        return {'state': state, 'progress': last_state.num_completed_tasks, 'total': last_op.num_outputs_total(), 'end_time': time.time() if state != 'RUNNING' else None, 'operators': {f'{op.name}{i}': {'progress': op_state.num_completed_tasks, 'total': op.num_outputs_total(), 'state': state} for i, (op, op_state) in enumerate(self._topology.items())}}

    def _update_stats_metrics(self, state: str, force_update: bool=False):
        StatsManager.update_execution_metrics(self._dataset_tag, [op.metrics for op in self._topology], self._get_operator_tags(), self._get_state_dict(state=state), force_update=force_update)