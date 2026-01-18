from numbers import Number, Integral
from .. import _api_internal
def _scalar_type_inference(value):
    if hasattr(value, 'dtype'):
        dtype = str(value.dtype)
    elif isinstance(value, bool):
        dtype = 'bool'
    elif isinstance(value, float):
        dtype = 'float32'
    elif isinstance(value, int):
        dtype = 'int32'
    else:
        raise NotImplementedError('Cannot automatically inference the type. value={}'.format(value))
    return dtype