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
def _get_bool_env_variable(name, default):
    val = os.environ.get(name)
    if val is None or len(val) == 0:
        return default
    try:
        return int(val) == 1
    except ValueError:
        return False