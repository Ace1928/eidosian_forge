import collections
import os
import threading
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.lib.io import tf_record
from tensorflow.python.util import compat
class Execution(ExecutionDigest):
    """Detailed data relating to a top-level execution event.

  The execution is of an individual op or a tf.function, which may have any
  number of output tensors.

  Properties (beyond the base class `ExecutionDigest`):
    host_name: Name of the host on which the execution happened.
    stack_frame_ids: Reference IDs for stack frames, ordered from bottommost to
      topmost. Use `DebugDataReader.read_execution_stack_trace()` to load the
      detailed stack frames (filepath, lineno and function name).
    tensor_debug_mode: TensorDebugMode enum value, as an `int`.
    graph_id: ID of the executed FuncGraph (applicable only the execution of a
      tf.function). `None` for the eager execution of an individual op.
    input_tensor_ids: IDs of the input (eager) tensor(s) for this execution, if
      any. If the eager execution has no input tensor, this is `None`. Else,
      this is a `tuple` of `int`s.
    output_tensor_ids: IDs of the output (eager) tensor(s) from this execution,
      if any. If the eager execution produces no output tensor, this is `None`.
      Else, this is a `tuple` of `int`s.
    debug_tensor_values: Values of the debug tensor(s), applicable only to
      non-FULL_TENSOR tensor debug mode. A tuple of list of numbers. Each
      element of the tuple corresponds to an output tensor of the execution.
      See documentation of the various TensorDebugModes for the semantics of the
      numbers. If the eager execution produces no output tensor, this is
      `None`. Else, this is a `tuple` of `list`s.
  """

    def __init__(self, execution_digest, host_name, stack_frame_ids, tensor_debug_mode, graph_id=None, input_tensor_ids=None, output_tensor_ids=None, debug_tensor_values=None):
        super().__init__(execution_digest.wall_time, execution_digest.locator, execution_digest.op_type, output_tensor_device_ids=execution_digest.output_tensor_device_ids)
        self._host_name = host_name
        self._stack_frame_ids = tuple(stack_frame_ids)
        self._tensor_debug_mode = tensor_debug_mode
        self._graph_id = graph_id
        self._input_tensor_ids = _tuple_or_none(input_tensor_ids)
        self._output_tensor_ids = _tuple_or_none(output_tensor_ids)
        self._debug_tensor_values = _tuple_or_none(debug_tensor_values)

    @property
    def host_name(self):
        return self._host_name

    @property
    def stack_frame_ids(self):
        return self._stack_frame_ids

    @property
    def tensor_debug_mode(self):
        return self._tensor_debug_mode

    @property
    def graph_id(self):
        return self._graph_id

    @property
    def input_tensor_ids(self):
        return self._input_tensor_ids

    @property
    def num_outputs(self):
        return len(self._output_tensor_ids) if self._output_tensor_ids else 0

    @property
    def output_tensor_ids(self):
        return self._output_tensor_ids

    @property
    def debug_tensor_values(self):
        return self._debug_tensor_values

    def to_json(self):
        output = super().to_json()
        output.update({'host_name': self.host_name, 'stack_frame_ids': self.stack_frame_ids, 'tensor_debug_mode': self.tensor_debug_mode, 'graph_id': self.graph_id, 'input_tensor_ids': self.input_tensor_ids, 'output_tensor_ids': self.output_tensor_ids, 'debug_tensor_values': self.debug_tensor_values})
        return output