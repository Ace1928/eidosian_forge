from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_optional_ops
@ops.RegisterGradient('OptionalFromValue')
def _OptionalFromValueGrad(op, grad):
    return gen_optional_ops.optional_get_value(grad, [t.dtype for t in op.inputs], [t.shape for t in op.inputs])