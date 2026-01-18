import warnings
import numpy
import cupy
from cupy_backends.cuda.api import runtime
from cupy import _core
from cupy._core import internal
from cupyx.scipy.ndimage import _util
def _generate_nd_kernel(name, pre, found, post, mode, w_shape, int_type, offsets, cval, ctype='X', preamble='', options=(), has_weights=True, has_structure=False, has_mask=False, binary_morphology=False, all_weights_nonzero=False):
    ndim = len(w_shape)
    in_params = 'raw X x'
    if has_weights:
        in_params += ', raw W w'
    if has_structure:
        in_params += ', raw S s'
    if has_mask:
        in_params += ', raw M mask'
    out_params = 'Y y'
    mode = 'grid-wrap' if mode == 'wrap' else mode
    size = '%s xsize_{j}=x.shape()[{j}], ysize_{j} = _raw_y.shape()[{j}], xstride_{j}=x.strides()[{j}];' % int_type
    sizes = [size.format(j=j) for j in range(ndim)]
    inds = _util._generate_indices_ops(ndim, int_type, offsets)
    expr = ' + '.join(['ix_{}'.format(j) for j in range(ndim)])
    ws_init = ws_pre = ws_post = ''
    if has_weights or has_structure:
        ws_init = 'int iws = 0;'
        if has_structure:
            ws_pre = 'S sval = s[iws];\n'
        if has_weights:
            ws_pre += 'W wval = w[iws];\n'
            if not all_weights_nonzero:
                ws_pre += 'if (nonzero(wval))'
        ws_post = 'iws++;'
    loops = []
    for j in range(ndim):
        if w_shape[j] == 1:
            loops.append('{{ {type} ix_{j} = ind_{j} * xstride_{j};'.format(j=j, type=int_type))
        else:
            boundary = _util._generate_boundary_condition_ops(mode, 'ix_{}'.format(j), 'xsize_{}'.format(j), int_type)
            loops.append('\n    for (int iw_{j} = 0; iw_{j} < {wsize}; iw_{j}++)\n    {{\n        {type} ix_{j} = ind_{j} + iw_{j};\n        {boundary}\n        ix_{j} *= xstride_{j};\n        '.format(j=j, wsize=w_shape[j], boundary=boundary, type=int_type))
    value = '(*(X*)&data[{expr}])'.format(expr=expr)
    if mode == 'constant':
        cond = ' || '.join(['(ix_{} < 0)'.format(j) for j in range(ndim)])
    if cval is numpy.nan:
        cval = 'CUDART_NAN'
    elif cval == numpy.inf:
        cval = 'CUDART_INF'
    elif cval == -numpy.inf:
        cval = '-CUDART_INF'
    if binary_morphology:
        found = found.format(cond=cond, value=value)
    else:
        if mode == 'constant':
            value = '(({cond}) ? cast<{ctype}>({cval}) : {value})'.format(cond=cond, ctype=ctype, cval=cval, value=value)
        found = found.format(value=value)
    operation = "\n    {sizes}\n    {inds}\n    // don't use a CArray for indexing (faster to deal with indexing ourselves)\n    const unsigned char* data = (const unsigned char*)&x[0];\n    {ws_init}\n    {pre}\n    {loops}\n        // inner-most loop\n        {ws_pre} {{\n            {found}\n        }}\n        {ws_post}\n    {end_loops}\n    {post}\n    ".format(sizes='\n'.join(sizes), inds=inds, pre=pre, post=post, ws_init=ws_init, ws_pre=ws_pre, ws_post=ws_post, loops='\n'.join(loops), found=found, end_loops='}' * ndim)
    mode_str = mode.replace('-', '_')
    name = 'cupyx_scipy_ndimage_{}_{}d_{}_w{}'.format(name, ndim, mode_str, '_'.join(['{}'.format(x) for x in w_shape]))
    if all_weights_nonzero:
        name += '_all_nonzero'
    if int_type == 'ptrdiff_t':
        name += '_i64'
    if has_structure:
        name += '_with_structure'
    if has_mask:
        name += '_with_mask'
    preamble = includes + _CAST_FUNCTION + preamble
    options += ('--std=c++11', '-DCUPY_USE_JITIFY')
    return cupy.ElementwiseKernel(in_params, out_params, operation, name, reduce_dims=False, preamble=preamble, options=options)