import collections
import os
import threading
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.lib.io import tf_record
from tensorflow.python.util import compat
class ExecutionDigest(BaseDigest):
    """Light-weight digest summarizing top-level execution event.

  Use `DebugDataReader.read_execution(execution_digest)` to load the more
  detailed data object concerning the execution event (`Execution`).

  Properties:
    op_type: Type name of the executed op. In the case of the eager execution of
      an individual op, it is the name of the op (e.g., "MatMul").
      In the case of the execution of a tf.function (FuncGraph), this is the
      internally-generated name of the function (e.g.,
      "__inference_my_func_123").
    output_tensor_device_ids: IDs of the devices on which the output tensors of
      the execution reside. For no-output execution, this is `None`.
  """

    def __init__(self, wall_time, locator, op_type, output_tensor_device_ids=None):
        super().__init__(wall_time, locator)
        self._op_type = op_type
        self._output_tensor_device_ids = _tuple_or_none(output_tensor_device_ids)

    @property
    def op_type(self):
        return self._op_type

    @property
    def output_tensor_device_ids(self):
        return self._output_tensor_device_ids

    def to_json(self):
        output = super().to_json()
        output.update({'op_type': self.op_type, 'output_tensor_device_ids': self.output_tensor_device_ids})
        return output