from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import ops
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import nest
def _convert_resource_variable_to_tensor(x):
    if _pywrap_utils.IsResourceVariable(x):
        return ops.convert_to_tensor(x)
    elif isinstance(x, composite_tensor.CompositeTensor):
        return composite_tensor.convert_variables_to_tensors(x)
    else:
        return x