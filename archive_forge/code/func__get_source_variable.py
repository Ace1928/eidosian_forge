import os
import sys
import ctypes
from numba.misc.findlib import find_lib
from numba.cuda.cuda_paths import get_cuda_paths
from numba.cuda.cudadrv.driver import locate_driver_and_loader, load_driver
from numba.cuda.cudadrv.error import CudaSupportError
def _get_source_variable(lib, static=False):
    if lib == 'nvvm':
        return get_cuda_paths()['nvvm'].by
    elif lib == 'libdevice':
        return get_cuda_paths()['libdevice'].by
    else:
        dir_type = 'static_cudalib_dir' if static else 'cudalib_dir'
        return get_cuda_paths()[dir_type].by