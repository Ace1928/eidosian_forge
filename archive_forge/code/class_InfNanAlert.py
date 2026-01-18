import numpy as np
from tensorflow.core.protobuf import debug_event_pb2
class InfNanAlert(object):
    """Alert for Infinity and NaN values."""

    def __init__(self, wall_time, op_type, output_slot, size=None, num_neg_inf=None, num_pos_inf=None, num_nan=None, execution_index=None, graph_execution_trace_index=None):
        self._wall_time = wall_time
        self._op_type = op_type
        self._output_slot = output_slot
        self._size = size
        self._num_neg_inf = num_neg_inf
        self._num_pos_inf = num_pos_inf
        self._num_nan = num_nan
        self._execution_index = execution_index
        self._graph_execution_trace_index = graph_execution_trace_index

    @property
    def wall_time(self):
        return self._wall_time

    @property
    def op_type(self):
        return self._op_type

    @property
    def output_slot(self):
        return self._output_slot

    @property
    def size(self):
        return self._size

    @property
    def num_neg_inf(self):
        return self._num_neg_inf

    @property
    def num_pos_inf(self):
        return self._num_pos_inf

    @property
    def num_nan(self):
        return self._num_nan

    @property
    def execution_index(self):
        return self._execution_index

    @property
    def graph_execution_trace_index(self):
        return self._graph_execution_trace_index