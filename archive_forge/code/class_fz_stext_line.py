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
class fz_stext_line(object):
    """	A text line is a list of characters that share a common baseline."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    wmode = property(_mupdf.fz_stext_line_wmode_get, _mupdf.fz_stext_line_wmode_set)
    dir = property(_mupdf.fz_stext_line_dir_get, _mupdf.fz_stext_line_dir_set)
    bbox = property(_mupdf.fz_stext_line_bbox_get, _mupdf.fz_stext_line_bbox_set)
    first_char = property(_mupdf.fz_stext_line_first_char_get, _mupdf.fz_stext_line_first_char_set)
    last_char = property(_mupdf.fz_stext_line_last_char_get, _mupdf.fz_stext_line_last_char_set)
    prev = property(_mupdf.fz_stext_line_prev_get, _mupdf.fz_stext_line_prev_set)
    next = property(_mupdf.fz_stext_line_next_get, _mupdf.fz_stext_line_next_set)

    def __init__(self):
        _mupdf.fz_stext_line_swiginit(self, _mupdf.new_fz_stext_line())
    __swig_destroy__ = _mupdf.delete_fz_stext_line