import sys
import re
import os
from collections import namedtuple
from numba.core.config import IS_WIN32
from numba.misc.findlib import find_lib, find_file
def _cudalib_path():
    if IS_WIN32:
        return 'bin'
    else:
        return 'lib64'