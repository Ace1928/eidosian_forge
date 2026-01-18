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
def _fix_roots(self, roots):
    roots = _convert_to_str_and_clear_empty(roots)
    new_roots = []
    for root in roots:
        path = self._absolute_normalized_path(root)
        if pydevd_constants.IS_WINDOWS:
            new_roots.append(path + '\\')
        else:
            new_roots.append(path + '/')
    return new_roots