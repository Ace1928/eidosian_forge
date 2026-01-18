import ctypes
import json
import os
import os.path
import shutil
import sys
from typing import Any, Dict, Optional
import warnings
def _setup_win32_dll_directory():
    if sys.platform.startswith('win32'):
        is_conda = os.environ.get('CONDA_PREFIX') is not None or os.environ.get('CONDA_BUILD_STATE') is not None
        cuda_path = get_cuda_path()
        if cuda_path is not None:
            if is_conda:
                cuda_bin_path = cuda_path
            else:
                cuda_bin_path = os.path.join(cuda_path, 'bin')
        else:
            cuda_bin_path = None
            warnings.warn('CUDA path could not be detected. Set CUDA_PATH environment variable if CuPy fails to load.')
        _log('CUDA_PATH: {}'.format(cuda_path))
        wheel_libdir = os.path.join(get_cupy_install_path(), 'cupy', '.data', 'lib')
        if os.path.isdir(wheel_libdir):
            _log('Wheel shared libraries: {}'.format(wheel_libdir))
        else:
            _log('Not wheel distribution ({} not found)'.format(wheel_libdir))
            wheel_libdir = None
        if (3, 8) <= sys.version_info:
            if cuda_bin_path is not None:
                _log('Adding DLL search path: {}'.format(cuda_bin_path))
                os.add_dll_directory(cuda_bin_path)
            if wheel_libdir is not None:
                _log('Adding DLL search path: {}'.format(wheel_libdir))
                os.add_dll_directory(wheel_libdir)
        elif wheel_libdir is not None:
            _log('Adding to PATH: {}'.format(wheel_libdir))
            path = os.environ.get('PATH', '')
            os.environ['PATH'] = wheel_libdir + os.pathsep + path