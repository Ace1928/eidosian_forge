from functools import partial
import itertools
import os
import sys
import socket as socket_module
from _pydev_bundle._pydev_imports_tipper import TYPE_IMPORT, TYPE_CLASS, TYPE_FUNCTION, TYPE_ATTR, \
from _pydev_bundle.pydev_is_thread_alive import is_thread_alive
from _pydev_bundle.pydev_override import overrides
from _pydevd_bundle._debug_adapter import pydevd_schema
from _pydevd_bundle._debug_adapter.pydevd_schema import ModuleEvent, ModuleEventBody, Module, \
from _pydevd_bundle.pydevd_comm_constants import CMD_THREAD_CREATE, CMD_RETURN, CMD_MODULE_EVENT, \
from _pydevd_bundle.pydevd_constants import get_thread_id, ForkSafeLock, DebugInfoHolder
from _pydevd_bundle.pydevd_net_command import NetCommand, NULL_NET_COMMAND
from _pydevd_bundle.pydevd_net_command_factory_xml import NetCommandFactory
from _pydevd_bundle.pydevd_utils import get_non_pydevd_threads
import pydevd_file_utils
from _pydevd_bundle.pydevd_comm import build_exception_info_response
from _pydevd_bundle.pydevd_additional_thread_info import set_additional_thread_info
from _pydevd_bundle import pydevd_frame_utils, pydevd_constants, pydevd_utils
import linecache
from io import StringIO
from _pydev_bundle import pydev_log
def _format_frame_name(self, fmt, initial_name, module_name, line, path):
    if fmt is None:
        return initial_name
    frame_name = initial_name
    if fmt.get('module', False):
        if module_name:
            if initial_name == '<module>':
                frame_name = module_name
            else:
                frame_name = '%s.%s' % (module_name, initial_name)
        else:
            basename = os.path.basename(path)
            basename = basename[0:-3] if basename.lower().endswith('.py') else basename
            if initial_name == '<module>':
                frame_name = '%s in %s' % (initial_name, basename)
            else:
                frame_name = '%s.%s' % (basename, initial_name)
    if fmt.get('line', False):
        frame_name = '%s : %d' % (frame_name, line)
    return frame_name