from ..block import HybridBlock
from ... import symbol
from ...base import numeric_types
from .activations import Activation
from ...util import is_np_array
def _infer_weight_shape(op_name, data_shape, kwargs):
    data = symbol.var('data', shape=data_shape)
    if is_np_array():
        op = getattr(symbol.npx, op_name)
        data = data.as_np_ndarray()
    else:
        op = getattr(symbol, op_name)
    sym = op(data, **kwargs)
    return sym.infer_shape_partial()[0]