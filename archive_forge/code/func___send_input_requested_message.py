import os
import sys
import traceback
from _pydev_bundle.pydev_imports import xmlrpclib, _queue, Exec
from  _pydev_bundle._pydev_calltip_util import get_description
from _pydevd_bundle import pydevd_vars
from _pydevd_bundle import pydevd_xml
from _pydevd_bundle.pydevd_constants import (IS_JYTHON, NEXT_VALUE_SEPARATOR, get_global_debugger,
from contextlib import contextmanager
from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_utils import interrupt_main_thread
from io import StringIO
def __send_input_requested_message(self, is_started):
    try:
        py_db = self._py_db
        if py_db is None:
            py_db = get_global_debugger()
        if py_db is None:
            return
        cmd = py_db.cmd_factory.make_input_requested_message(is_started)
        py_db.writer.add_command(cmd)
    except Exception:
        pydev_log.exception()