import numpy
import cupy
from cupy import _core
from cupy._core import internal
from cupyx.scipy.ndimage import _util
from cupyx.scipy.ndimage import _filters_core
from cupyx.scipy.ndimage import _filters_generic
@cupy._util.memoize(for_each_device=True)
def _get_min_or_max_kernel(mode, w_shape, func, offsets, cval, int_type, has_weights=True, has_structure=False, has_central_value=True):
    ctype = 'X' if has_weights else 'double'
    value = '{value}'
    if not has_weights:
        value = 'cast<double>({})'.format(value)
    if has_structure:
        value += ('-' if func == 'min' else '+') + 'cast<X>(sval)'
    if has_central_value:
        pre = '{} value = x[i];'
        found = 'value = {func}({value}, value);'
    else:
        pre = '{} value; bool set = false;'
        found = 'value = set ? {func}({value}, value) : {value}; set=true;'
    return _filters_core._generate_nd_kernel(func, pre.format(ctype), found.format(func=func, value=value), 'y = cast<Y>(value);', mode, w_shape, int_type, offsets, cval, ctype=ctype, has_weights=has_weights, has_structure=has_structure)