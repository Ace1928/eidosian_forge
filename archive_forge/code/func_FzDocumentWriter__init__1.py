from sys import version_info as _swig_python_version_info
import weakref
import inspect
import os
import re
import sys
import traceback
import inspect
import io
import os
import sys
import traceback
import types
def FzDocumentWriter__init__1(self, *args):
    out = None
    for arg in args:
        if isinstance(arg, FzOutput2):
            assert not out, 'More than one FzOutput2 passed to FzDocumentWriter.__init__()'
            out = arg
    if out:
        self._out = out
    return FzDocumentWriter__init__0(self, *args)