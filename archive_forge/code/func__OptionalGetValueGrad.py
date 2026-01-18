from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_optional_ops
@ops.RegisterGradient('OptionalGetValue')
def _OptionalGetValueGrad(unused_op, *grads):
    return gen_optional_ops.optional_from_value(grads)