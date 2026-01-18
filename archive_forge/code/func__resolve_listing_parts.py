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
def _resolve_listing_parts(resolved, parts_in_lowercase, filename):
    try:
        if parts_in_lowercase == ['']:
            return resolved
        return _resolve_listing(resolved, iter(parts_in_lowercase))
    except FileNotFoundError:
        _listdir_cache.clear()
        try:
            return _resolve_listing(resolved, iter(parts_in_lowercase))
        except FileNotFoundError:
            if os_path_exists(filename):
                pydev_log.critical('pydev debugger: critical: unable to get real case for file. Details:\nfilename: %s\ndrive: %s\nparts: %s\n(please create a ticket in the tracker to address this).', filename, resolved, parts_in_lowercase)
                pydev_log.exception()
            return filename
    except OSError:
        if DebugInfoHolder.DEBUG_TRACE_LEVEL >= 1:
            pydev_log.info('pydev debugger: OSError: Unable to get real case for file. Details:\nfilename: %s\ndrive: %s\nparts: %s\n', filename, resolved, parts_in_lowercase)
            pydev_log.exception()
        return filename