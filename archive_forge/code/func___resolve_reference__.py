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
def __resolve_reference__(self, text):
    """

        :type text: str
        """
    obj = None
    if '.' not in text:
        try:
            obj = self.get_namespace()[text]
        except KeyError:
            pass
        if obj is None:
            try:
                obj = self.get_namespace()['__builtins__'][text]
            except:
                pass
        if obj is None:
            try:
                obj = getattr(self.get_namespace()['__builtins__'], text, None)
            except:
                pass
    else:
        try:
            last_dot = text.rindex('.')
            parent_context = text[0:last_dot]
            res = pydevd_vars.eval_in_context(parent_context, self.get_namespace(), self.get_namespace())
            obj = getattr(res, text[last_dot + 1:])
        except:
            pass
    return obj