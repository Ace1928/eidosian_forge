import sys
import re
import os
from collections import namedtuple
from numba.core.config import IS_WIN32
from numba.misc.findlib import find_lib, find_file
def _get_static_cudalib_dir():
    by, libdir = _get_static_cudalib_dir_path_decision()
    return _env_path_tuple(by, libdir)