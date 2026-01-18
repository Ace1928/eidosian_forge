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
def _original_file_to_client(filename, cache={}):
    try:
        return cache[filename]
    except KeyError:
        translated = _path_to_expected_str(get_path_with_real_case(absolute_path(filename)))
        cache[filename] = (translated, False)
    return cache[filename]