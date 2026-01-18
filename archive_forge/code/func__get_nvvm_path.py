import sys
import re
import os
from collections import namedtuple
from numba.core.config import IS_WIN32
from numba.misc.findlib import find_lib, find_file
def _get_nvvm_path():
    by, path = _get_nvvm_path_decision()
    candidates = find_lib('nvvm', path)
    path = max(candidates) if candidates else None
    return _env_path_tuple(by, path)