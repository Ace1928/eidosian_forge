from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg_impl as _linalg
def _GetGradWrt(output_grad, other_operand, input_shape, input_subs, other_subs, output_subs):
    """Returns the gradient wrt an input operand for a binary einsum.

    This function does not handle (un)broadcasting. This must be done separately
    on the returned gradient.

    Args:
      output_grad: The gradient wrt the output of a binary einsum operation.
      other_operand: The complementary `Tensor` operand i.e. which is not the
        input operand.
      input_shape: A `Tensor` representing the shape of input operand.
      input_subs: The subscripts of the input operand.
      other_subs: The subscripts of the complementary operand.
      output_subs: The output subscripts.
    """
    reduced_label_set = set(input_subs).difference(set(output_subs + other_subs + '.'))
    left_subs = ''.join((s for s in input_subs if s not in reduced_label_set))
    grad_reduced = gen_linalg_ops.einsum([output_grad, other_operand], '{},{}->{}'.format(output_subs, other_subs, left_subs))
    if not reduced_label_set:
        return grad_reduced
    return _GetGradReduced(grad_reduced, left_subs, input_subs, input_shape, reduced_label_set)