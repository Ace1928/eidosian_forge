import warnings
from tensorflow.python.autograph.core import ag_ctx as autograph_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.data.ops import debug_mode
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.ops import script_ops
from tensorflow.python.util import function_utils
from tensorflow.python.util import variable_utils
def _should_pack(arg):
    """Determines whether the caller needs to pack the argument in a tuple.

  If user-defined function returns a list of tensors, `nest.flatten()` and
  `ops.convert_to_tensor()` and would conspire to attempt to stack those tensors
  into a single tensor because the tf.data version of `nest.flatten()` does
  not recurse into lists. Since it is more likely that the list arose from
  returning the result of an operation (such as `tf.numpy_function()`) that
  returns a list of not-necessarily-stackable tensors, we treat the returned
  value as a `tuple` instead. A user wishing to pack the return value into a
  single tensor can use an explicit `tf.stack()` before returning.

  Args:
    arg: argument to check

  Returns:
    Indication of whether the caller needs to pack the argument in a tuple.
  """
    return isinstance(arg, list)