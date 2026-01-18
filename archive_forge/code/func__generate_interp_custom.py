import numpy
import cupy
import cupy._core.internal
from cupyx.scipy.ndimage import _spline_prefilter_core
from cupyx.scipy.ndimage import _spline_kernel_weights
from cupyx.scipy.ndimage import _util
def _generate_interp_custom(coord_func, ndim, large_int, yshape, mode, cval, order, name='', integer_output=False, nprepad=0, omit_in_coord=False):
    """
    Args:
        coord_func (function): generates code to do the coordinate
            transformation. See for example, `_get_coord_shift`.
        ndim (int): The number of dimensions.
        large_int (bool): If true use Py_ssize_t instead of int for indexing.
        yshape (tuple): Shape of the output array.
        mode (str): Signal extension mode to use at the array boundaries
        cval (float): constant value used when `mode == 'constant'`.
        name (str): base name for the interpolation kernel
        integer_output (bool): boolean indicating whether the output has an
            integer type.
        nprepad (int): integer indicating the amount of prepadding at the
            boundaries.

    Returns:
        operation (str): code body for the ElementwiseKernel
        name (str): name for the ElementwiseKernel
    """
    ops = []
    internal_dtype = 'double' if integer_output else 'Y'
    ops.append(f'{internal_dtype} out = 0.0;')
    if large_int:
        uint_t = 'size_t'
        int_t = 'ptrdiff_t'
    else:
        uint_t = 'unsigned int'
        int_t = 'int'
    for j in range(ndim):
        ops.append(f'const {int_t} xsize_{j} = x.shape()[{j}];')
    ops.append(f'const {uint_t} sx_{ndim - 1} = 1;')
    for j in range(ndim - 1, 0, -1):
        ops.append(f'const {uint_t} sx_{j - 1} = sx_{j} * xsize_{j};')
    if not omit_in_coord:
        ops.append(_unravel_loop_index(yshape, uint_t))
    ops = ops + coord_func(ndim, nprepad)
    if cval is numpy.nan:
        cval = '(Y)CUDART_NAN'
    elif cval == numpy.inf:
        cval = '(Y)CUDART_INF'
    elif cval == -numpy.inf:
        cval = '(Y)(-CUDART_INF)'
    else:
        cval = f'({internal_dtype}){cval}'
    if mode == 'constant':
        _cond = ' || '.join([f'(c_{j} < 0) || (c_{j} > xsize_{j} - 1)' for j in range(ndim)])
        ops.append(f'\n        if ({_cond})\n        {{\n            out = {cval};\n        }}\n        else\n        {{')
    if order == 0:
        if mode == 'wrap':
            ops.append('double dcoord;')
        for j in range(ndim):
            if mode == 'wrap':
                ops.append(f'\n                dcoord = c_{j};')
            else:
                ops.append(f'\n                {int_t} cf_{j} = ({int_t})floor((double)c_{j} + 0.5);')
            if mode != 'constant':
                if mode == 'wrap':
                    ixvar = 'dcoord'
                    float_ix = True
                else:
                    ixvar = f'cf_{j}'
                    float_ix = False
                ops.append(_util._generate_boundary_condition_ops(mode, ixvar, f'xsize_{j}', int_t, float_ix))
                if mode == 'wrap':
                    ops.append(f'\n                {int_t} cf_{j} = ({int_t})floor(dcoord + 0.5);')
            ops.append(f'\n            {int_t} ic_{j} = cf_{j} * sx_{j};')
        _coord_idx = ' + '.join([f'ic_{j}' for j in range(ndim)])
        if mode == 'grid-constant':
            _cond = ' || '.join([f'(ic_{j} < 0)' for j in range(ndim)])
            ops.append(f'\n            if ({_cond}) {{\n                out = {cval};\n            }} else {{\n                out = ({internal_dtype})x[{_coord_idx}];\n            }}')
        else:
            ops.append(f'\n            out = ({internal_dtype})x[{_coord_idx}];')
    elif order == 1:
        for j in range(ndim):
            ops.append(f'\n            {int_t} cf_{j} = ({int_t})floor((double)c_{j});\n            {int_t} cc_{j} = cf_{j} + 1;\n            {int_t} n_{j} = (c_{j} == cf_{j}) ? 1 : 2;  // points needed\n            ')
            if mode == 'wrap':
                ops.append(f'\n                double dcoordf = c_{j};\n                double dcoordc = c_{j} + 1;')
            else:
                ops.append(f'\n                {int_t} cf_bounded_{j} = cf_{j};\n                {int_t} cc_bounded_{j} = cc_{j};')
            if mode != 'constant':
                if mode == 'wrap':
                    ixvar = 'dcoordf'
                    float_ix = True
                else:
                    ixvar = f'cf_bounded_{j}'
                    float_ix = False
                ops.append(_util._generate_boundary_condition_ops(mode, ixvar, f'xsize_{j}', int_t, float_ix))
                ixvar = 'dcoordc' if mode == 'wrap' else f'cc_bounded_{j}'
                ops.append(_util._generate_boundary_condition_ops(mode, ixvar, f'xsize_{j}', int_t, float_ix))
                if mode == 'wrap':
                    ops.append(f'\n                    {int_t} cf_bounded_{j} = ({int_t})floor(dcoordf);;\n                    {int_t} cc_bounded_{j} = ({int_t})floor(dcoordf + 1);;\n                    ')
            ops.append(f'\n            for (int s_{j} = 0; s_{j} < n_{j}; s_{j}++)\n                {{\n                    W w_{j};\n                    {int_t} ic_{j};\n                    if (s_{j} == 0)\n                    {{\n                        w_{j} = (W)cc_{j} - c_{j};\n                        ic_{j} = cf_bounded_{j} * sx_{j};\n                    }} else\n                    {{\n                        w_{j} = c_{j} - (W)cf_{j};\n                        ic_{j} = cc_bounded_{j} * sx_{j};\n                    }}')
    elif order > 1:
        if mode == 'grid-constant':
            spline_mode = 'constant'
        elif mode == 'nearest':
            spline_mode = 'nearest'
        else:
            spline_mode = _spline_prefilter_core._get_spline_mode(mode)
        ops.append(f'\n            W wx, wy;\n            {int_t} start;')
        for j in range(ndim):
            ops.append(f'\n            W weights_{j}[{order + 1}];')
            ops.append(spline_weights_inline[order].format(j=j, order=order))
            if mode in ['wrap']:
                ops.append(f'double dcoord = c_{j};')
                coord_var = 'dcoord'
                ops.append(_util._generate_boundary_condition_ops(mode, coord_var, f'xsize_{j}', int_t, True))
            else:
                coord_var = f'(double)c_{j}'
            if order & 1:
                op_str = '\n                start = ({int_t})floor({coord_var}) - {order_2};'
            else:
                op_str = '\n                start = ({int_t})floor({coord_var} + 0.5) - {order_2};'
            ops.append(op_str.format(int_t=int_t, coord_var=coord_var, order_2=order // 2))
            ops.append(f'{int_t} ci_{j}[{order + 1}];')
            for k in range(order + 1):
                ixvar = f'ci_{j}[{k}]'
                ops.append(f'\n                {ixvar} = start + {k};')
                ops.append(_util._generate_boundary_condition_ops(spline_mode, ixvar, f'xsize_{j}', int_t))
            ops.append(f'\n            W w_{j};\n            {int_t} ic_{j};\n            for (int k_{j} = 0; k_{j} <= {order}; k_{j}++)\n                {{\n                    w_{j} = weights_{j}[k_{j}];\n                    ic_{j} = ci_{j}[k_{j}] * sx_{j};\n            ')
    if order > 0:
        _weight = ' * '.join([f'w_{j}' for j in range(ndim)])
        _coord_idx = ' + '.join([f'ic_{j}' for j in range(ndim)])
        if mode == 'grid-constant' or (order > 1 and mode == 'constant'):
            _cond = ' || '.join([f'(ic_{j} < 0)' for j in range(ndim)])
            ops.append(f'\n            if ({_cond}) {{\n                out += {cval} * ({internal_dtype})({_weight});\n            }} else {{\n                {internal_dtype} val = ({internal_dtype})x[{_coord_idx}];\n                out += val * ({internal_dtype})({_weight});\n            }}')
        else:
            ops.append(f'\n            {internal_dtype} val = ({internal_dtype})x[{_coord_idx}];\n            out += val * ({internal_dtype})({_weight});')
        ops.append('}' * ndim)
    if mode == 'constant':
        ops.append('}')
    if integer_output:
        ops.append('y = (Y)rint((double)out);')
    else:
        ops.append('y = (Y)out;')
    operation = '\n'.join(ops)
    mode_str = mode.replace('-', '_')
    name = 'cupyx_scipy_ndimage_interpolate_{}_order{}_{}_{}d_y{}'.format(name, order, mode_str, ndim, '_'.join([f'{j}' for j in yshape]))
    if uint_t == 'size_t':
        name += '_i64'
    return (operation, name)