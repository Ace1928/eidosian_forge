import math
import cupy
from cupy._core import internal
from cupyx.scipy import fft
from cupyx.scipy.ndimage import _filters
from cupyx.scipy.ndimage import _util
def _direct_correlate(in1, in2, mode='full', output=float, convolution=False, boundary='constant', fillvalue=0.0, shift=False):
    if in1.ndim != 1 and (in1.dtype.kind == 'b' or (in1.dtype.kind == 'f' and in1.dtype.itemsize < 4)):
        raise ValueError('unsupported type in SciPy')
    swapped_inputs = False
    orig_in1_shape = in1.shape
    if _inputs_swap_needed(mode, in1.shape, in2.shape) or (in2.size > in1.size and boundary == 'constant' and (fillvalue == 0)):
        in1, in2 = (in2, in1)
        swapped_inputs = True
    if in2.nbytes >= 1 << 31:
        raise RuntimeError('smaller array must be 2 GiB or less, use method="fft" instead')
    if mode == 'full':
        out_shape = tuple((x1 + x2 - 1 for x1, x2 in zip(in1.shape, in2.shape)))
        offsets = tuple((x - 1 for x in in2.shape))
    elif mode == 'valid':
        out_shape = tuple((x1 - x2 + 1 for x1, x2 in zip(in1.shape, in2.shape)))
        offsets = (0,) * in1.ndim
    else:
        out_shape = orig_in1_shape
        if orig_in1_shape == in1.shape:
            offsets = tuple(((x - shift) // 2 for x in in2.shape))
        else:
            offsets = tuple(((2 * x2 - x1 - (not convolution) + shift) // 2 for x1, x2 in zip(in1.shape, in2.shape)))
    out_dtype = cupy.promote_types(in1, in2)
    if not isinstance(output, cupy.ndarray):
        if not cupy.can_cast(output, out_dtype):
            raise ValueError('not available for this type')
        output = cupy.empty(out_shape, out_dtype)
    elif output.shape != out_shape:
        raise ValueError('out has wrong shape')
    elif output.dtype != out_dtype:
        raise ValueError('out has wrong dtype')
    if cupy.can_cast(in2, in1):
        in2 = in2.astype(out_dtype)
    int_type = _util._get_inttype(in1)
    kernel = _filters._get_correlate_kernel(boundary, in2.shape, int_type, offsets, fillvalue)
    in2 = _reverse(in2) if convolution else in2.conj()
    if not swapped_inputs or convolution:
        kernel(in1, in2, output)
    elif output.dtype.kind != 'c':
        kernel(in1, in2, _reverse(output))
    else:
        kernel(in1, in2, output)
        output = cupy.ascontiguousarray(_reverse(output))
        if swapped_inputs and (mode != 'valid' or not shift):
            cupy.conjugate(output, out=output)
    return output