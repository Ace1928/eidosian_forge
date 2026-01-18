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
@_util.memoize(for_each_device=True)
def _get_arch():
    nvrtc_max_compute_capability = _get_max_compute_capability()
    arch = device.Device().compute_capability
    if arch in _tegra_archs:
        return arch
    else:
        return min(arch, nvrtc_max_compute_capability)