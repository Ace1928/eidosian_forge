import collections
import hashlib
import os
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tensor_tracer_pb2
def _write_trace_points(self, tensor_trace_points):
    """Writes the list of checkpoints."""
    self._write_report('%s %s\n' % (_MARKER_SECTION_BEGIN, _SECTION_NAME_TENSOR_TRACER_CHECKPOINT))
    for tensor, checkpoint_name in tensor_trace_points:
        self._write_report('%s %s\n' % (tensor.name, checkpoint_name))
    self._write_report('%s %s\n' % (_MARKER_SECTION_END, _SECTION_NAME_TENSOR_TRACER_CHECKPOINT))