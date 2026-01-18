import re
import numpy
import cupy
import cupy._core._routines_manipulation as _manipulation
from cupy._core._dtype import get_dtype, _raise_if_invalid_cast
from cupy._core import internal
def _get_args_transposed(self, args, input_axes, outs, output_axes):
    transposed_args = []
    missing_dims = set()
    for i, (arg, iax, input_coredims, md) in enumerate(zip(args, input_axes, self._input_coredimss, self._min_dims)):
        shape = arg.shape
        nds = len(shape)
        if nds < md:
            raise ValueError(f'Input operand {i} does not have enough dimensions (has {nds}, gufunc core with signature {self._signature} requires {md}')
        optionals = len(input_coredims) - nds
        if optionals > 0:
            if input_coredims[0][-1] == '?':
                shape = (1,) * optionals + shape
                missing_dims.update(set(input_coredims[:optionals]))
            else:
                shape = shape + (1,) * optionals
                missing_dims.update(set(input_coredims[min(0, len(shape) - 1):]))
            arg = arg.reshape(shape)
        transposed_args.append(self._transpose_element(arg, iax, shape))
    args = transposed_args
    if outs is not None:
        transposed_outs = []
        for out, iox, coredims in zip(outs, output_axes, self._output_coredimss):
            transposed_outs.append(self._transpose_element(out, iox, out.shape))
        if len(transposed_outs) == len(outs):
            outs = transposed_outs
    shape = internal._broadcast_shapes([a.shape[:-len(self._input_coredimss)] for a in args])
    args = [_manipulation.broadcast_to(a, shape + a.shape[-len(self._input_coredimss):]) for a in args]
    input_shapes = [a.shape for a in args]
    num_loopdims = [len(s) - len(cd) for s, cd in zip(input_shapes, self._input_coredimss)]
    max_loopdims = max(num_loopdims) if num_loopdims else None
    core_input_shapes = [dict(zip(icd, s[n:])) for s, n, icd in zip(input_shapes, num_loopdims, self._input_coredimss)]
    core_shapes = {}
    for d in core_input_shapes:
        core_shapes.update(d)
    loop_input_dimss = [tuple(('__loopdim%d__' % d for d in range(max_loopdims - n, max_loopdims))) for n in num_loopdims]
    input_dimss = [li + c for li, c in zip(loop_input_dimss, self._input_coredimss)]
    loop_output_dims = max(loop_input_dimss, key=len, default=())
    dimsizess = {}
    for dims, shape in zip(input_dimss, input_shapes):
        for dim, size in zip(dims, shape):
            dimsizes = dimsizess.get(dim, [])
            dimsizes.append(size)
            dimsizess[dim] = dimsizes
    for dim, sizes in dimsizess.items():
        if set(sizes).union({1}) != {1, max(sizes)}:
            raise ValueError(f'Dimension {dim} with different lengths in arrays')
    return (args, dimsizess, loop_output_dims, outs, missing_dims)