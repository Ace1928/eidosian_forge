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
class FzErrorSyntax(FzErrorBase):
    """ For `FZ_ERROR_SYNTAX`."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def __init__(self, message):
        _mupdf.FzErrorSyntax_swiginit(self, _mupdf.new_FzErrorSyntax(message))
    __swig_destroy__ = _mupdf.delete_FzErrorSyntax