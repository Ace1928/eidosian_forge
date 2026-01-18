import sys
import re
import os
from collections import namedtuple
from numba.core.config import IS_WIN32
from numba.misc.findlib import find_lib, find_file
def _get_cudalib_dir_path_decision():
    options = [('Conda environment', get_conda_ctk()), ('Conda environment (NVIDIA package)', get_nvidia_cudalib_ctk()), ('CUDA_HOME', get_cuda_home(_cudalib_path())), ('System', get_system_ctk(_cudalib_path()))]
    by, libdir = _find_valid_path(options)
    return (by, libdir)