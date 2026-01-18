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
def _preprocess_hipcc(source, options):
    cmd = ['hipcc', '--preprocess'] + list(options)
    with tempfile.TemporaryDirectory() as root_dir:
        path = os.path.join(root_dir, 'kern')
        cu_path = '%s.cpp' % path
        with open(cu_path, 'w') as cu_file:
            cu_file.write(source)
        cmd.append(cu_path)
        pp_src = _run_cc(cmd, root_dir, 'hipcc')
        assert isinstance(pp_src, str)
        return re.sub('(?m)^#.*$', '', pp_src)