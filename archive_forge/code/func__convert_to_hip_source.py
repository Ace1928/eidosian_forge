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
def _convert_to_hip_source(source, extra_source, is_hiprtc):
    if not is_hiprtc:
        return '#include <hip/hip_runtime.h>\n' + source
    if _cuda_hip_version >= 40400000:
        return source
    if _cuda_hip_version >= 402:
        return '#include <hip/hip_runtime.h>\n' + source
    global _hip_extra_source
    if _hip_extra_source is None:
        if extra_source is not None:
            extra_source = extra_source.split('\n')
            extra_source = [line for line in extra_source if not line.startswith('#include') and (not line.startswith('#pragma once'))]
            _hip_extra_source = extra_source = '\n'.join(extra_source)
    source = source.split('\n')
    source = [line for line in source if not line.startswith('#include')]
    source = '#include <hip/hip_runtime.h>\n#include <hip/hip_fp16.h>\n' + _hip_extra_source + '\n'.join(source)
    return source