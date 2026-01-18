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
def _get_path_with_real_case(filename):
    if '~' in filename:
        filename = convert_to_long_pathname(filename)
    if filename.startswith('<') or not os_path_exists(filename):
        return filename
    drive, parts = os.path.splitdrive(os.path.normpath(filename))
    drive = drive.upper()
    while parts.startswith(os.path.sep):
        parts = parts[1:]
        drive += os.path.sep
    parts = parts.lower().split(os.path.sep)
    return _resolve_listing_parts(drive, parts, filename)