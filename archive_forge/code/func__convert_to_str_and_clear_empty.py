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
def _convert_to_str_and_clear_empty(roots):
    new_roots = []
    for root in roots:
        assert isinstance(root, str), '%s not str (found: %s)' % (root, type(root))
        if root:
            new_roots.append(root)
    return new_roots