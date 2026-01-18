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
def _abs_and_canonical_path(filename, NORM_PATHS_CONTAINER=NORM_PATHS_CONTAINER):
    try:
        return NORM_PATHS_CONTAINER[filename]
    except:
        if filename.__class__ != str:
            raise AssertionError('Paths passed to _abs_and_canonical_path must be str. Found: %s (%s)' % (filename, type(filename)))
        if os is None:
            return (filename, filename)
        os_path = os.path
        if os_path is None:
            return (filename, filename)
        os_path_abspath = os_path.abspath
        os_path_isabs = os_path.isabs
        if os_path_abspath is None or os_path_isabs is None or os_path_real_path is None:
            return (filename, filename)
        isabs = os_path_isabs(filename)
        if _global_resolve_symlinks:
            os_path_abspath = os_path_real_path
        normalize = False
        abs_path = _apply_func_and_normalize_case(filename, os_path_abspath, isabs, normalize)
        normalize = True
        real_path = _apply_func_and_normalize_case(filename, os_path_real_path, isabs, normalize)
        NORM_PATHS_CONTAINER[filename] = (abs_path, real_path)
        return (abs_path, real_path)