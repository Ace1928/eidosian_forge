from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.checkpoint import checkpoint_options
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.saved_model import registration
from tensorflow.python.trackable import trackable_utils
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
def _get_mapped_registered_restore_fn(fn, trackables, call_with_mapped_captures):
    """Converts the function to a python or tf.function with a single file arg."""

    def restore_fn(merged_prefix):
        return fn(trackables=trackables, merged_prefix=merged_prefix)
    if call_with_mapped_captures is None:
        return restore_fn
    else:
        tf_fn = def_function.function(restore_fn, autograph=False)
        concrete = tf_fn.get_concrete_function(merged_prefix=tensor_spec.TensorSpec(shape=(), dtype=dtypes.string))

        def restore_fn_with_replaced_captures(merged_prefix):
            return call_with_mapped_captures(concrete, [merged_prefix])
        return restore_fn_with_replaced_captures