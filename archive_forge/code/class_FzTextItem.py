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
class FzTextItem(object):
    """
    Wrapper class for struct `fz_text_item`. Not copyable or assignable.
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

    def __init__(self, *args):
        """
        *Overload 1:*
        Default constructor, sets `m_internal` to null.

        |

        *Overload 2:*
        Constructor using raw copy of pre-existing `::fz_text_item`.
        """
        _mupdf.FzTextItem_swiginit(self, _mupdf.new_FzTextItem(*args))
    __swig_destroy__ = _mupdf.delete_FzTextItem

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzTextItem_m_internal_value(self)
    m_internal = property(_mupdf.FzTextItem_m_internal_get, _mupdf.FzTextItem_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzTextItem_s_num_instances_get, _mupdf.FzTextItem_s_num_instances_set)