import numpy as np
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import weak_tensor
from tensorflow.python.ops import variables
from tensorflow.python.util import nest
def _result_type_impl(*arrays_and_dtypes):
    """Internal implementation of jnp_style_result_type.

  Args:
    *arrays_and_dtypes: A list of Tensors, Variables, NumPy arrays or python
      numbers.

  Returns:
    The result promotion type from all the inputs.

  Raises:
    TypeError: when the promotion between the input dtypes is disabled in the
    current mode

    NotImplementedError:
      (1) When arrays_and_dtypes contains an unsupported input type (e.g.
      RaggedTensor).
      (2) When there isn't a possible promotion for the input dtypes.
  """
    promo_safety_mode = ops.get_dtype_conversion_mode()
    valid_arrays_and_dtypes = []
    for inp in arrays_and_dtypes:
        if inp is not None:
            if _is_acceptable_input_type(inp):
                valid_arrays_and_dtypes.append(inp)
            else:
                raise NotImplementedError(f'Auto dtype conversion semantics does not support {type(inp)} type.')
    dtypes_and_is_weak = [_get_dtype_and_weakness(x) for x in nest.flatten(valid_arrays_and_dtypes)]
    if not dtypes_and_is_weak:
        dtypes_and_is_weak = [(dtypes.float32, True)]
    res = dtypes_and_is_weak[0]
    for arg in dtypes_and_is_weak[1:]:
        res = (res[0].base_dtype, res[1])
        arg = (arg[0].base_dtype, arg[1])
        try:
            res_next, allowed_mode = _BINARY_DTYPE_RES_FULL[res][arg]
        except KeyError as exc:
            raise NotImplementedError(f'Implicit Conversion between {res[0]} and {arg[0]} is not allowed. Please convert the input manually if you need to.') from exc
        if allowed_mode.value > promo_safety_mode.value:
            raise TypeError(f'In promotion mode {promo_safety_mode}, implicit dtype promotion between ({res[0]}, weak={res[1]}) and ({arg[0]}, weak={arg[1]}) is disallowed. You need to explicitly specify the dtype in your op, or relax your dtype promotion rules (such as from SAFE mode to ALL mode).')
        res = res_next
    return res