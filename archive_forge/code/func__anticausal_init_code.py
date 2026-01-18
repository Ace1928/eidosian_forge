import functools
import math
import operator
import textwrap
import cupy
def _anticausal_init_code(mode):
    """Code for the anti-causal initialization step of IIR filtering.

    c is a 1d array of length n and z is a filter pole
    """
    code = f'\n        // anti-causal init for mode={mode}'
    if mode == 'mirror':
        code += '\n        c[(n - 1) * element_stride] = (\n            z * c[(n - 2) * element_stride] +\n            c[(n - 1) * element_stride]) * z / (z * z - 1);'
    elif mode == 'grid-wrap':
        code += '\n        z_i = z;\n\n        for (i = 0; i < min(n - 1, static_cast<idx_t>({n_boundary})); ++i) {{\n            c[(n - 1) * element_stride] += z_i * c[i * element_stride];\n            z_i *= z;\n        }}\n        c[(n - 1) * element_stride] *= z / (z_i - 1); /* z_i = pow(z, n) */'
    elif mode == 'reflect':
        code += '\n        c[(n - 1) * element_stride] *= z / (z - 1);'
    else:
        raise ValueError('invalid mode: {}'.format(mode))
    return code