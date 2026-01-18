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
def _map_file_to_server(filename, cache=norm_filename_to_server_container):
    try:
        return cache[filename]
    except KeyError:
        if eclipse_sep != python_sep:
            filename = filename.replace(python_sep, eclipse_sep)
        translated = filename
        translated_normalized = _normcase_from_client(filename)
        for eclipse_prefix, server_prefix in paths_from_eclipse_to_python:
            if translated_normalized.startswith(eclipse_prefix):
                found_translation = True
                if DEBUG_CLIENT_SERVER_TRANSLATION:
                    pydev_log.critical('pydev debugger: replacing to server: %s', filename)
                translated = server_prefix + filename[len(eclipse_prefix):]
                if DEBUG_CLIENT_SERVER_TRANSLATION:
                    pydev_log.critical('pydev debugger: sent to server: %s - matched prefix: %s', translated, eclipse_prefix)
                break
        else:
            found_translation = False
        if eclipse_sep != python_sep:
            translated = translated.replace(eclipse_sep, python_sep)
        if found_translation:
            translated = absolute_path(translated)
        elif not os_path_exists(translated):
            if not translated.startswith('<'):
                error_once('pydev debugger: unable to find translation for: "%s" in [%s] (please revise your path mappings).\n', filename, ', '.join(['"%s"' % (x[0],) for x in paths_from_eclipse_to_python]))
        else:
            translated = absolute_path(translated)
        cache[filename] = translated
        return translated