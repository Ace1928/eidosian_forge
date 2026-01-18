import ctypes
import json
import os
import os.path
import shutil
import sys
from typing import Any, Dict, Optional
import warnings
def _get_hipcc_path():
    rocm_path = get_rocm_path()
    if rocm_path is None:
        return None
    return shutil.which('hipcc', path=os.path.join(rocm_path, 'bin'))