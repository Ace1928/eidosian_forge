from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_functional_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.gen_functional_ops import remote_call
from tensorflow.python.ops.gen_functional_ops import symbolic_gradient
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def If(cond, inputs, then_branch, else_branch, name=None):
    """output = Cond(inputs) ?

  then_branch(inputs) : else_branch(inputs).

  Args:
    cond: A `Tensor`. A scalar. If the scalar is not a boolean, the scalar is
      converted to a boolean according to the following rule: if the scalar is a
        numerical value, non-zero means True and zero means False; if the scalar
        is a string, non-empty means True and empty means False.
    inputs: A list of input tensors.
    then_branch: A function takes 'inputs' and returns a list of tensors, whose
      types are the same as what else_branch returns.
    else_branch: A function takes 'inputs' and returns a list of tensors. whose
      types are the same as what then_branch returns.
    name: A name for the operation (optional).

  Returns:
    A list of tensors returned by either then_branch(inputs)
    or else_branch(inputs).
  """
    if isinstance(then_branch, function._DefinedFunction):
        tlist = [_.type for _ in then_branch.definition.signature.output_arg]
        return gen_functional_ops._if(cond, inputs, tlist, then_branch, else_branch, name=name)
    then_out = then_branch.structured_outputs
    else_out = else_branch.structured_outputs
    nest.assert_same_structure(then_out, else_out, expand_composites=True)
    tlist = nest.flatten(then_branch.output_dtypes)
    ret = gen_functional_ops._if(cond, inputs, tlist, then_branch, else_branch, name=name)
    return nest.pack_sequence_as(then_out, ret, expand_composites=True)