from _pydev_bundle._pydev_saved_modules import threading
from _pydev_bundle import _pydev_saved_modules
from _pydevd_bundle.pydevd_utils import notify_about_gevent_if_needed
import weakref
from _pydevd_bundle.pydevd_constants import IS_JYTHON, IS_IRONPYTHON, \
from _pydev_bundle.pydev_log import exception as pydev_log_exception
import sys
from _pydev_bundle import pydev_log
import pydevd_tracing
from _pydevd_bundle.pydevd_collect_bytecode_info import iter_instructions
def _collect_load_names(func):
    found_load_names = set()
    for instruction in iter_instructions(func.__code__):
        if instruction.opname in ('LOAD_GLOBAL', 'LOAD_ATTR', 'LOAD_METHOD'):
            found_load_names.add(instruction.argrepr)
    return found_load_names