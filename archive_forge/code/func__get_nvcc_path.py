import ctypes
import json
import os
import os.path
import shutil
import sys
from typing import Any, Dict, Optional
import warnings
def _get_nvcc_path():
    nvcc_path = os.environ.get('NVCC', None)
    if nvcc_path is not None:
        return nvcc_path
    cuda_path = get_cuda_path()
    if cuda_path is None:
        return None
    return shutil.which('nvcc', path=os.path.join(cuda_path, 'bin'))