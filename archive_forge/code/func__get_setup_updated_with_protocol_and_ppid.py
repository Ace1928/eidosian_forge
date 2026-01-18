import os
import re
import sys
from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle.pydevd_constants import get_global_debugger, IS_WINDOWS, IS_JYTHON, get_current_thread_id, \
from _pydev_bundle import pydev_log
from contextlib import contextmanager
from _pydevd_bundle import pydevd_constants, pydevd_defaults
from _pydevd_bundle.pydevd_defaults import PydevdCustomization
import ast
def _get_setup_updated_with_protocol_and_ppid(setup, is_exec=False):
    if setup is None:
        setup = {}
    setup = setup.copy()
    setup.pop(pydevd_constants.ARGUMENT_HTTP_JSON_PROTOCOL, None)
    setup.pop(pydevd_constants.ARGUMENT_JSON_PROTOCOL, None)
    setup.pop(pydevd_constants.ARGUMENT_QUOTED_LINE_PROTOCOL, None)
    if not is_exec:
        setup[pydevd_constants.ARGUMENT_PPID] = os.getpid()
    protocol = pydevd_constants.get_protocol()
    if protocol == pydevd_constants.HTTP_JSON_PROTOCOL:
        setup[pydevd_constants.ARGUMENT_HTTP_JSON_PROTOCOL] = True
    elif protocol == pydevd_constants.JSON_PROTOCOL:
        setup[pydevd_constants.ARGUMENT_JSON_PROTOCOL] = True
    elif protocol == pydevd_constants.QUOTED_LINE_PROTOCOL:
        setup[pydevd_constants.ARGUMENT_QUOTED_LINE_PROTOCOL] = True
    elif protocol == pydevd_constants.HTTP_PROTOCOL:
        setup[pydevd_constants.ARGUMENT_HTTP_PROTOCOL] = True
    else:
        pydev_log.debug('Unexpected protocol: %s', protocol)
    mode = pydevd_defaults.PydevdCustomization.DEBUG_MODE
    if mode:
        setup['debug-mode'] = mode
    preimport = pydevd_defaults.PydevdCustomization.PREIMPORT
    if preimport:
        setup['preimport'] = preimport
    if DebugInfoHolder.PYDEVD_DEBUG_FILE:
        setup['log-file'] = DebugInfoHolder.PYDEVD_DEBUG_FILE
    if DebugInfoHolder.DEBUG_TRACE_LEVEL:
        setup['log-level'] = DebugInfoHolder.DEBUG_TRACE_LEVEL
    return setup