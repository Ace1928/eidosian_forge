import fnmatch
import glob
import os.path
import sys
from _pydev_bundle import pydev_log
import pydevd_file_utils
import json
from collections import namedtuple
from _pydev_bundle._pydev_saved_modules import threading
from pydevd_file_utils import normcase
from _pydevd_bundle.pydevd_constants import USER_CODE_BASENAMES_STARTING_WITH, \
from _pydevd_bundle import pydevd_constants
def _check_matches(patterns, paths):
    if not patterns and (not paths):
        return True
    if not patterns and paths or (patterns and (not paths)):
        return False
    pattern = normcase(patterns[0])
    path = normcase(paths[0])
    if not glob.has_magic(pattern):
        if pattern != path:
            return False
    elif pattern == '**':
        if len(patterns) == 1:
            return True
        for i in range(len(paths)):
            if _check_matches(patterns[1:], paths[i:]):
                return True
    elif not fnmatch.fnmatch(path, pattern):
        return False
    return _check_matches(patterns[1:], paths[1:])