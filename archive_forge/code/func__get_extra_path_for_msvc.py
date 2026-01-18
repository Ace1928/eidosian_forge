import copy
import hashlib
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from cupy.cuda import device
from cupy.cuda import function
from cupy.cuda import get_rocm_path
from cupy_backends.cuda.api import driver
from cupy_backends.cuda.api import runtime
from cupy_backends.cuda.libs import nvrtc
from cupy import _util
@_util.memoize()
def _get_extra_path_for_msvc():
    cl_exe = shutil.which('cl.exe')
    if cl_exe:
        return None
    try:
        import setuptools
        vctools = setuptools.msvc.EnvironmentInfo(platform.machine()).VCTools
    except Exception as e:
        warnings.warn(f'Failed to auto-detect cl.exe path: {type(e)}: {e}')
        return None
    for path in vctools:
        cl_exe = os.path.join(path, 'cl.exe')
        if os.path.exists(cl_exe):
            return path
    warnings.warn(f'cl.exe could not be found in {vctools}')
    return None