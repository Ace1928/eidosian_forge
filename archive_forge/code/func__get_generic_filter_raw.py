import cupy
from cupy_backends.cuda.api import runtime
from cupy import _util
from cupyx.scipy.ndimage import _filters_core
@_util.memoize(for_each_device=True)
def _get_generic_filter_raw(rk, filter_size, mode, wshape, offsets, cval, int_type):
    """Generic filter implementation based on a raw kernel."""
    setup = '\n    int iv = 0;\n    double values[{}];\n    double val_out;'.format(filter_size)
    sub_call = 'raw_kernel::{}(values, {}, &val_out);\n    y = cast<Y>(val_out);'.format(rk.name, filter_size)
    return _filters_core._generate_nd_kernel('generic_{}_{}'.format(filter_size, rk.name), setup, 'values[iv++] = cast<double>({value});', sub_call, mode, wshape, int_type, offsets, cval, preamble='namespace raw_kernel {{\n{}\n}}'.format(rk.code.replace('__global__', '__device__')), options=rk.options)