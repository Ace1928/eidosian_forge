import numpy as np
from tensorflow.core.protobuf import debug_event_pb2
class InfNanMonitor(BaseMonitor):
    """Monitor for Infinity and NaN in tensor values."""

    def __init__(self, debug_events_reader, limit=0):
        super(InfNanMonitor, self).__init__(debug_events_reader)
        self._limit = limit
        self._alerts = []

    def _check_full_tensor_value(self, tensor_value, wall_time, op_type, output_slot, execution_index=None, graph_execution_trace_index=None):
        """Check a full tensor value.

    Appends to the list of alerts if any inf or nan is found in the full tensor
    value.

    Args:
      tensor_value: The full tensor value as a `np.ndarray`.
      wall_time: Wall timestamp for the execution event that generated the
        tensor value.
      op_type: Op type executed.
      output_slot: The output slot of the op.
      execution_index: Index to the top-level execution event.
      graph_execution_trace_index: Index to the intra-graph execution trace
        (if applicable.)
    """
        size = np.size(tensor_value)
        if not size or not np.issubdtype(tensor_value.dtype, np.floating):
            return
        is_inf = np.isinf(tensor_value)
        num_neg_inf = np.count_nonzero(np.logical_and(is_inf, np.less(tensor_value, 0.0)))
        num_pos_inf = np.count_nonzero(np.logical_and(is_inf, np.greater(tensor_value, 0.0)))
        num_nan = np.count_nonzero(np.isnan(tensor_value))
        if num_neg_inf or num_pos_inf or num_nan:
            self._alerts.append(InfNanAlert(wall_time, op_type, output_slot, size=size, num_neg_inf=num_neg_inf, num_pos_inf=num_pos_inf, num_nan=num_nan, execution_index=execution_index, graph_execution_trace_index=graph_execution_trace_index))

    def _check_debug_tensor_value(self, tensor_debug_mode, debug_tensor_value, wall_time, op_type, output_slot, execution_index=None, graph_execution_trace_index=None):
        """Check for bad numerical values based on debug summary of tensor value.

    If tensor_debug_mode is one in which debug_tensor_value does not carry
    information about the presence or count of inf / nan values (e.g., SHAPE),
    this method is a no-op.

    When infs and/or nans are found, `InfNanAlert` objects are created and
    appended to `self._alerts`.

    Args:
      tensor_debug_mode: TensorDebugMode proto enum.
      debug_tensor_value: Debug tensor value as a list of numbers.
      wall_time: Wall timestamp for the tensor event.
      op_type: Type of the op that generated the tensor (e.g., "Conv2D").
      output_slot: Output slot index of the tensor for the op.
      execution_index: Top-level execution index.
      graph_execution_trace_index: Intra-graph execution index.
    """
        assert tensor_debug_mode != debug_event_pb2.TensorDebugMode.FULL_TENSOR
        if not debug_tensor_value:
            return
        if tensor_debug_mode == debug_event_pb2.TensorDebugMode.CURT_HEALTH:
            _, any_nan_inf = debug_tensor_value
            if any_nan_inf:
                self._alerts.append(InfNanAlert(wall_time, op_type, output_slot, execution_index=execution_index, graph_execution_trace_index=graph_execution_trace_index))
        elif tensor_debug_mode == debug_event_pb2.TensorDebugMode.CONCISE_HEALTH:
            _, size, num_neg_inf, num_pos_inf, num_nan = debug_tensor_value
            if num_neg_inf or num_pos_inf or num_nan:
                self._alerts.append(InfNanAlert(wall_time, op_type, output_slot, size=size, num_neg_inf=num_neg_inf, num_pos_inf=num_pos_inf, num_nan=num_nan, execution_index=execution_index, graph_execution_trace_index=graph_execution_trace_index))
        elif tensor_debug_mode == debug_event_pb2.TensorDebugMode.FULL_HEALTH:
            _, _, _, _, size, num_neg_inf, num_pos_inf, num_nan, _, _, _ = debug_tensor_value
            if num_neg_inf or num_pos_inf or num_nan:
                self._alerts.append(InfNanAlert(wall_time, op_type, output_slot, size=size, num_neg_inf=num_neg_inf, num_pos_inf=num_pos_inf, num_nan=num_nan, execution_index=execution_index, graph_execution_trace_index=graph_execution_trace_index))

    def on_execution(self, execution_index, execution):
        if self._limit > 0 and len(self._alerts) >= self._limit:
            return
        if execution.tensor_debug_mode == debug_event_pb2.TensorDebugMode.FULL_TENSOR:
            tensor_values = self._debug_data_reader.execution_to_tensor_values(execution)
            for output_slot, tensor_value in enumerate(tensor_values):
                self._check_full_tensor_value(tensor_value, execution.wall_time, execution.op_type, output_slot, execution_index=execution_index)
        elif execution.debug_tensor_values:
            for output_slot, debug_tensor_value in enumerate(execution.debug_tensor_values):
                self._check_debug_tensor_value(execution.tensor_debug_mode, debug_tensor_value, execution.wall_time, execution.op_type, output_slot, execution_index=execution_index)

    def on_graph_execution_trace(self, graph_execution_trace_index, graph_execution_trace):
        """Monitor method for GraphExecutionTrace data object."""
        if self._limit > 0 and len(self._alerts) >= self._limit:
            return
        if graph_execution_trace.tensor_debug_mode == debug_event_pb2.TensorDebugMode.FULL_TENSOR:
            tensor_value = self._debug_data_reader.graph_execution_trace_to_tensor_value(graph_execution_trace)
            self._check_full_tensor_value(tensor_value, graph_execution_trace.wall_time, graph_execution_trace.op_type, graph_execution_trace.output_slot, graph_execution_trace_index=graph_execution_trace_index)
        elif graph_execution_trace.debug_tensor_value:
            self._check_debug_tensor_value(graph_execution_trace.tensor_debug_mode, graph_execution_trace.debug_tensor_value, graph_execution_trace.wall_time, graph_execution_trace.op_type, graph_execution_trace.output_slot, graph_execution_trace_index=graph_execution_trace_index)

    def alerts(self):
        return self._alerts