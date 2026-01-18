import warnings
import numpy
import cupy
def _generate_boundary_condition_ops(mode, ix, xsize, int_t='int', float_ix=False):
    min_func = 'fmin' if float_ix else 'min'
    max_func = 'fmax' if float_ix else 'max'
    if mode in ['reflect', 'grid-mirror']:
        ops = '\n        if ({ix} < 0) {{\n            {ix} = - 1 -{ix};\n        }}\n        {ix} %= {xsize} * 2;\n        {ix} = {min}({ix}, 2 * {xsize} - 1 - {ix});'.format(ix=ix, xsize=xsize, min=min_func)
    elif mode == 'mirror':
        ops = '\n        if ({xsize} == 1) {{\n            {ix} = 0;\n        }} else {{\n            if ({ix} < 0) {{\n                {ix} = -{ix};\n            }}\n            {ix} = 1 + ({ix} - 1) % (({xsize} - 1) * 2);\n            {ix} = {min}({ix}, 2 * {xsize} - 2 - {ix});\n        }}'.format(ix=ix, xsize=xsize, min=min_func)
    elif mode == 'nearest':
        ops = '\n        {ix} = {min}({max}(({T}){ix}, ({T})0), ({T})({xsize} - 1));'.format(ix=ix, xsize=xsize, min=min_func, max=max_func, T='int' if int_t == 'int' else 'long long')
    elif mode == 'grid-wrap':
        ops = '\n        {ix} %= {xsize};\n        while ({ix} < 0) {{\n            {ix} += {xsize};\n        }}'.format(ix=ix, xsize=xsize)
    elif mode == 'wrap':
        ops = '\n        if ({ix} < 0) {{\n            {ix} += ({sz} - 1) * (({int_t})(-{ix} / ({sz} - 1)) + 1);\n        }} else if ({ix} > ({sz} - 1)) {{\n            {ix} -= ({sz} - 1) * ({int_t})({ix} / ({sz} - 1));\n        }};'.format(ix=ix, sz=xsize, int_t=int_t)
    elif mode in ['constant', 'grid-constant']:
        ops = '\n        if (({ix} < 0) || {ix} >= {xsize}) {{\n            {ix} = -1;\n        }}'.format(ix=ix, xsize=xsize)
    return ops