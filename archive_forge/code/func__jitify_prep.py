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
def _jitify_prep(source, options, cu_path):
    global _jitify_header_source_map_populated
    if not _jitify_header_source_map_populated:
        from cupy._core import core
        jitify._add_sources(core._get_header_source_map())
        _jitify_header_source_map_populated = True
    old_source = source
    source = cu_path + '\n' + source
    try:
        name, options, headers, include_names = jitify.jitify(source, options)
    except Exception as e:
        cex = CompileException(str(e), old_source, cu_path, options, 'jitify')
        dump = _get_bool_env_variable('CUPY_DUMP_CUDA_SOURCE_ON_ERROR', False)
        if dump:
            cex.dump(sys.stderr)
        raise JitifyException(str(cex))
    assert name == cu_path
    return (options, headers, include_names)