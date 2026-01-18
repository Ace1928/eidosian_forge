import sys
import re
import os
from collections import namedtuple
from numba.core.config import IS_WIN32
from numba.misc.findlib import find_lib, find_file
def _get_nvvm_path_decision():
    options = [('Conda environment', get_conda_ctk()), ('Conda environment (NVIDIA package)', get_nvidia_nvvm_ctk()), ('CUDA_HOME', get_cuda_home(*_nvvm_lib_dir())), ('System', get_system_ctk(*_nvvm_lib_dir()))]
    by, path = _find_valid_path(options)
    return (by, path)