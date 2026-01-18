import contextlib
import warnings
from google.protobuf import json_format as _json_format
from tensorflow.core.framework.summary_pb2 import Summary
from tensorflow.core.framework.summary_pb2 import SummaryDescription
from tensorflow.core.framework.summary_pb2 import SummaryMetadata as _SummaryMetadata  # pylint: enable=unused-import
from tensorflow.core.util.event_pb2 import Event
from tensorflow.core.util.event_pb2 import SessionLog
from tensorflow.core.util.event_pb2 import TaggedRunMetadata
from tensorflow.python.distribute import summary_op_util as _distribute_summary_op_util
from tensorflow.python.eager import context as _context
from tensorflow.python.framework import constant_op as _constant_op
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import ops as _ops
from tensorflow.python.ops import array_ops as _array_ops
from tensorflow.python.ops import gen_logging_ops as _gen_logging_ops
from tensorflow.python.ops import gen_summary_ops as _gen_summary_ops  # pylint: disable=unused-import
from tensorflow.python.ops import summary_op_util as _summary_op_util
from tensorflow.python.ops import summary_ops_v2 as _summary_ops_v2
from tensorflow.python.summary.writer.writer import FileWriter
from tensorflow.python.summary.writer.writer_cache import FileWriterCache
from tensorflow.python.training import training_util as _training_util
from tensorflow.python.util import compat as _compat
from tensorflow.python.util.tf_export import tf_export
def _should_invoke_v2_op():
    """Check if v2 op can be invoked.

  When calling TF1 summary op in eager mode, if the following conditions are
  met, v2 op will be invoked:
  - The outermost context is eager mode.
  - A default TF2 summary writer is present.
  - A step is set for the writer (using `tf.summary.SummaryWriter.as_default`,
    `tf.summary.experimental.set_step` or
    `tf.compat.v1.train.create_global_step`).

  Returns:
    A boolean indicating whether v2 summary op should be invoked.
  """
    if not _ops.executing_eagerly_outside_functions():
        return False
    if not _summary_ops_v2.has_default_writer():
        warnings.warn('Cannot activate TF2 compatibility support for TF1 summary ops: default summary writer not found.')
        return False
    if _get_step_for_v2() is None:
        warnings.warn('Cannot activate TF2 compatibility support for TF1 summary ops: global step not set. To set step for summary writer, use `tf.summary.SummaryWriter.as_default(step=_)`, `tf.summary.experimental.set_step()` or `tf.compat.v1.train.create_global_step()`.')
        return False
    return True