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
class fz_arc4(object):
    """
    Structure definition is public to enable stack
    based allocation. Do not access the members directly.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    x = property(_mupdf.fz_arc4_x_get, _mupdf.fz_arc4_x_set)
    y = property(_mupdf.fz_arc4_y_get, _mupdf.fz_arc4_y_set)
    state = property(_mupdf.fz_arc4_state_get, _mupdf.fz_arc4_state_set)

    def __init__(self):
        _mupdf.fz_arc4_swiginit(self, _mupdf.new_fz_arc4())
    __swig_destroy__ = _mupdf.delete_fz_arc4