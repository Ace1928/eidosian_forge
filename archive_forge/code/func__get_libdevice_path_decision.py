import sys
import re
import os
from collections import namedtuple
from numba.core.config import IS_WIN32
from numba.misc.findlib import find_lib, find_file
def _get_libdevice_path_decision():
    options = [('Conda environment', get_conda_ctk()), ('Conda environment (NVIDIA package)', get_nvidia_libdevice_ctk()), ('CUDA_HOME', get_cuda_home('nvvm', 'libdevice')), ('System', get_system_ctk('nvvm', 'libdevice')), ('Debian package', get_debian_pkg_libdevice())]
    by, libdir = _find_valid_path(options)
    return (by, libdir)