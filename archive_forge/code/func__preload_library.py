import ctypes
import json
import os
import os.path
import shutil
import sys
from typing import Any, Dict, Optional
import warnings
def _preload_library(lib):
    """Preload dependent shared libraries.

    The preload configuration file (cupy/.data/_wheel.json) will be added
    during the wheel build process.
    """
    _log(f'Preloading triggered for library: {lib}')
    if not _can_attempt_preload(lib):
        return
    _preload_libs[lib] = {}
    config = get_preload_config()
    cuda_version = config['cuda']
    _log('CuPy wheel package built for CUDA {}'.format(cuda_version))
    cupy_cuda_lib_path = get_cupy_cuda_lib_path()
    _log('CuPy CUDA library directory: {}'.format(cupy_cuda_lib_path))
    version = config[lib]['version']
    filenames = config[lib]['filenames']
    for filename in filenames:
        _log(f'Looking for {lib} version {version} ({filename})')
        libpath_cands = [os.path.join(cupy_cuda_lib_path, config['cuda'], lib, version, x, filename) for x in ['lib', 'lib64', 'bin']]
        for libpath in libpath_cands:
            if not os.path.exists(libpath):
                _log('Rejected candidate (not found): {}'.format(libpath))
                continue
            try:
                _log(f'Trying to load {libpath}')
                _preload_libs[lib][libpath] = ctypes.CDLL(libpath)
                _log('Loaded')
                break
            except Exception as e:
                e_type = type(e).__name__
                msg = f'CuPy failed to preload library ({libpath}): {e_type} ({e})'
                _log(msg)
                warnings.warn(msg)
        else:
            _log('File {} could not be found'.format(filename))
            _log(f'Trying to load {filename} from default search path')
            try:
                _preload_libs[lib][filename] = ctypes.CDLL(filename)
                _log('Loaded')
            except Exception as e:
                _log(f'Library {lib} could not be preloaded: {e}')