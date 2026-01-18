import functools
import math
import operator
import textwrap
import cupy
def _get_spline1d_code(mode, poles, n_boundary):
    """Generates the code required for IIR filtering of a single 1d signal.

    Prefiltering is done by causal filtering followed by anti-causal filtering.
    Multiple boundary conditions have been implemented.
    """
    code = ['\n    __device__ void spline_prefilter1d(\n        T* __restrict__ c, idx_t signal_length, idx_t element_stride)\n    {{']
    code.append('\n        idx_t i, n = signal_length;\n        P z, z_i;')
    mode = _get_spline_mode(mode)
    if mode == 'mirror':
        code.append('\n        P z_n_1;')
    elif mode == 'reflect':
        code.append('\n        P z_n;\n        T c0;')
    for pole in poles:
        code.append(f'\n        // select the current pole\n        z = {pole};')
        code.append(_causal_init_code(mode))
        code.append('\n        // apply the causal filter for the current pole\n        for (i = 1; i < n; ++i) {{\n            c[i * element_stride] += z * c[(i - 1) * element_stride];\n        }}')
        code.append('\n        #ifdef __HIP_DEVICE_COMPILE__\n        __syncthreads();\n        #endif\n        ')
        code.append(_anticausal_init_code(mode))
        code.append('\n        // apply the anti-causal filter for the current pole\n        for (i = n - 2; i >= 0; --i) {{\n            c[i * element_stride] = z * (c[(i + 1) * element_stride] -\n                                         c[i * element_stride]);\n        }}')
    code += ['\n    }}']
    return textwrap.dedent('\n'.join(code)).format(n_boundary=n_boundary)