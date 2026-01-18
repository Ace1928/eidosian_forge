from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_constants import DebugInfoHolder, IS_WINDOWS, IS_JYTHON, \
from _pydev_bundle._pydev_filesystem_encoding import getfilesystemencoding
from _pydevd_bundle.pydevd_comm_constants import file_system_encoding, filesystem_encoding_is_utf8
from _pydev_bundle.pydev_log import error_once
import json
import os.path
import sys
import itertools
import ntpath
from functools import partial
def _get_relative_filename_abs_path(filename, func, os_path_exists=os_path_exists):
    for p in sys.path:
        r = func(os.path.join(p, filename))
        if os_path_exists(r):
            return r
    r = func(os.path.join(_library_dir, filename))
    return r