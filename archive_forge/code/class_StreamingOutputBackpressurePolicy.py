import time
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, Tuple
import ray
from .backpressure_policy import BackpressurePolicy
from ray.data._internal.dataset_logger import DatasetLogger
class StreamingOutputBackpressurePolicy(BackpressurePolicy):
    """A backpressure policy that throttles the streaming outputs of the `DataOpTask`s.

    The are 2 levels of configs to control the behavior:
    - At the Ray Core level, we use
      `MAX_BLOCKS_IN_GENERATOR_BUFFER` to limit the number of blocks buffered in
      the streaming generator of each OpDataTask. When it's reached, the task will
      be blocked at `yield` until the caller reads another `ObjectRef.
    - At the Ray Data level, we use
      `MAX_BLOCKS_IN_GENERATOR_BUFFER` to limit the number of blocks buffered in the
      output queue of each operator. When it's reached, we'll stop reading from the
      streaming generators of the op's tasks, and thus trigger backpressure at the
      Ray Core level.

    Thus, total number of buffered blocks for each operator can be
    `MAX_BLOCKS_IN_GENERATOR_BUFFER * num_running_tasks +
    MAX_BLOCKS_IN_OP_OUTPUT_QUEUE`.
    """
    MAX_BLOCKS_IN_GENERATOR_BUFFER = 10
    MAX_BLOCKS_IN_GENERATOR_BUFFER_CONFIG_KEY = 'backpressure_policies.streaming_output.max_blocks_in_generator_buffer'
    MAX_BLOCKS_IN_OP_OUTPUT_QUEUE = 20
    MAX_BLOCKS_IN_OP_OUTPUT_QUEUE_CONFIG_KEY = 'backpressure_policies.streaming_output.max_blocks_in_op_output_queue'
    MAX_OUTPUT_IDLE_SECONDS = 10

    def __init__(self, topology: 'Topology'):
        data_context = ray.data.DataContext.get_current()
        self._max_num_blocks_in_streaming_gen_buffer = data_context.get_config(self.MAX_BLOCKS_IN_GENERATOR_BUFFER_CONFIG_KEY, self.MAX_BLOCKS_IN_GENERATOR_BUFFER)
        assert self._max_num_blocks_in_streaming_gen_buffer > 0
        data_context._task_pool_data_task_remote_args['_generator_backpressure_num_objects'] = 2 * self._max_num_blocks_in_streaming_gen_buffer
        self._max_num_blocks_in_op_output_queue = data_context.get_config(self.MAX_BLOCKS_IN_OP_OUTPUT_QUEUE_CONFIG_KEY, self.MAX_BLOCKS_IN_OP_OUTPUT_QUEUE)
        assert self._max_num_blocks_in_op_output_queue > 0
        self._last_num_outputs_and_time: Dict['PhysicalOperator', Tuple[int, float]] = defaultdict(lambda: (0, time.time()))
        self._warning_printed = False

    def calculate_max_blocks_to_read_per_op(self, topology: 'Topology') -> Dict['OpState', int]:
        max_blocks_to_read_per_op: Dict['OpState', int] = {}
        downstream_idle = False
        for op, state in reversed(topology.items()):
            max_blocks_to_read_per_op[state] = self._max_num_blocks_in_op_output_queue - state.outqueue_num_blocks()
            if downstream_idle:
                max_blocks_to_read_per_op[state] = max(max_blocks_to_read_per_op[state], 1)
            downstream_idle = False
            if op.num_active_tasks() == 0:
                downstream_idle = True
            else:
                cur_num_outputs = state.op.metrics.num_outputs_generated
                cur_time = time.time()
                last_num_outputs, last_time = self._last_num_outputs_and_time[state.op]
                if cur_num_outputs > last_num_outputs:
                    self._last_num_outputs_and_time[state.op] = (cur_num_outputs, cur_time)
                elif cur_time - last_time > self.MAX_OUTPUT_IDLE_SECONDS:
                    downstream_idle = True
                    self._print_warning(state.op, cur_time - last_time)
        return max_blocks_to_read_per_op

    def _print_warning(self, op: 'PhysicalOperator', idle_time: float):
        if self._warning_printed:
            return
        self._warning_printed = True
        msg = f'Operator {op} is running but has no outputs for {idle_time} seconds. Execution may be slower than expected.\nIgnore this warning if your UDF is expected to be slow. Otherwise, this can happen when there are fewer cluster resources available to Ray Data than expected. If you have non-Data tasks or actors running in the cluster, exclude their resources from Ray Data with `DataContext.get_current().execution_options.exclude_resources`. This message will only print once.'
        logger.get_logger().warning(msg)