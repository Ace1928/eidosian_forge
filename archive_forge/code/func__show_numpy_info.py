imports at module scope, instead importing numpy within function calls.
import sys
import os
def _show_numpy_info():
    import numpy as np
    print('NumPy version %s' % np.__version__)
    relaxed_strides = np.ones((10, 1), order='C').flags.f_contiguous
    print('NumPy relaxed strides checking option:', relaxed_strides)
    info = np.lib.utils._opt_info()
    print('NumPy CPU features: ', info if info else 'nothing enabled')