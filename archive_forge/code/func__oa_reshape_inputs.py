import math
import cupy
from cupy._core import internal
from cupyx.scipy import fft
from cupyx.scipy.ndimage import _filters
from cupyx.scipy.ndimage import _util
def _oa_reshape_inputs(in1, in2, axes, shape_final, block_size, overlaps, in1_step, in2_step):
    nsteps1 = []
    nsteps2 = []
    pad_size1 = []
    pad_size2 = []
    for i in range(in1.ndim):
        if i not in axes:
            pad_size1 += [(0, 0)]
            pad_size2 += [(0, 0)]
            continue
        curnstep1, curpad1, curnstep2, curpad2 = (1, 0, 1, 0)
        if in1.shape[i] > in1_step[i]:
            curnstep1 = math.ceil((in1.shape[i] + 1) / in1_step[i])
            if (block_size[i] - overlaps[i]) * curnstep1 < shape_final[i]:
                curnstep1 += 1
            curpad1 = curnstep1 * in1_step[i] - in1.shape[i]
        if in2.shape[i] > in2_step[i]:
            curnstep2 = math.ceil((in2.shape[i] + 1) / in2_step[i])
            if (block_size[i] - overlaps[i]) * curnstep2 < shape_final[i]:
                curnstep2 += 1
            curpad2 = curnstep2 * in2_step[i] - in2.shape[i]
        nsteps1 += [curnstep1]
        nsteps2 += [curnstep2]
        pad_size1 += [(0, curpad1)]
        pad_size2 += [(0, curpad2)]
    if not all((curpad == (0, 0) for curpad in pad_size1)):
        in1 = cupy.pad(in1, pad_size1, mode='constant', constant_values=0)
    if not all((curpad == (0, 0) for curpad in pad_size2)):
        in2 = cupy.pad(in2, pad_size2, mode='constant', constant_values=0)
    reshape_size1 = list(in1_step)
    reshape_size2 = list(in2_step)
    for i, iax in enumerate(axes):
        reshape_size1.insert(iax + i, nsteps1[i])
        reshape_size2.insert(iax + i, nsteps2[i])
    return (in1.reshape(*reshape_size1), in2.reshape(*reshape_size2))