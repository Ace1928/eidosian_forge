from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_tpu_ops
from tensorflow.python.ops.gen_tpu_ops import *
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tpu_function
from tensorflow.python.util.tf_export import tf_export
@ops.RegisterGradient('CollectivePermute')
def _collective_permute_grad(op, grad):
    source_target_pairs = op.inputs[1][:, ::-1]
    return [gen_tpu_ops.collective_permute(grad, source_target_pairs), None]