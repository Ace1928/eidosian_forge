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
def _resolve_listing(resolved, iter_parts_lowercase, cache=_listdir_cache):
    while True:
        try:
            resolve_lowercase = next(iter_parts_lowercase)
        except StopIteration:
            return resolved
        resolved_lower = resolved.lower()
        resolved_joined = cache.get((resolved_lower, resolve_lowercase))
        if resolved_joined is None:
            dir_contents = cache.get(resolved_lower)
            if dir_contents is None:
                dir_contents = cache[resolved_lower] = os_listdir(resolved)
            for filename in dir_contents:
                if filename.lower() == resolve_lowercase:
                    resolved_joined = os.path.join(resolved, filename)
                    cache[resolved_lower, resolve_lowercase] = resolved_joined
                    break
            else:
                raise FileNotFoundError('Unable to find: %s in %s. Dir Contents: %s' % (resolve_lowercase, resolved, dir_contents))
        resolved = resolved_joined