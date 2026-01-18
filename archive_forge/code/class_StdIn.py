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
class StdIn(BaseStdIn):
    """
        Object to be added to stdin (to emulate it as non-blocking while the next line arrives)
    """

    def __init__(self, interpreter, host, client_port, original_stdin=sys.stdin):
        BaseStdIn.__init__(self, original_stdin)
        self.interpreter = interpreter
        self.client_port = client_port
        self.host = host

    def readline(self, *args, **kwargs):
        try:
            server = xmlrpclib.Server('http://%s:%s' % (self.host, self.client_port))
            requested_input = server.RequestInput()
            if not requested_input:
                return '\n'
            else:
                requested_input += '\n'
            return requested_input
        except KeyboardInterrupt:
            raise
        except:
            return '\n'

    def close(self, *args, **kwargs):
        pass