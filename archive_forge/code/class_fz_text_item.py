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
class fz_text_item(object):
    """
    Text buffer.

    The trm field contains the a, b, c and d coefficients.
    The e and f coefficients come from the individual elements,
    together they form the transform matrix for the glyph.

    Glyphs are referenced by glyph ID.
    The Unicode text equivalent is kept in a separate array
    with indexes into the glyph array.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    x = property(_mupdf.fz_text_item_x_get, _mupdf.fz_text_item_x_set)
    y = property(_mupdf.fz_text_item_y_get, _mupdf.fz_text_item_y_set)
    gid = property(_mupdf.fz_text_item_gid_get, _mupdf.fz_text_item_gid_set)
    ucs = property(_mupdf.fz_text_item_ucs_get, _mupdf.fz_text_item_ucs_set)
    cid = property(_mupdf.fz_text_item_cid_get, _mupdf.fz_text_item_cid_set)

    def __init__(self):
        _mupdf.fz_text_item_swiginit(self, _mupdf.new_fz_text_item())
    __swig_destroy__ = _mupdf.delete_fz_text_item