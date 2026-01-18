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
class FzTextSpan(object):
    """ Wrapper class for struct `fz_text_span`. Not copyable or assignable."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def font(self):
        """ Gives class-aware access to m_internal->font."""
        return _mupdf.FzTextSpan_font(self)

    def trm(self):
        """ Gives class-aware access to m_internal->trm."""
        return _mupdf.FzTextSpan_trm(self)

    def items(self, i):
        """
        Gives access to m_internal->items[i].
        						Returned reference is only valid as long as `this`.
        						Provided mainly for use by SWIG bindings.
        """
        return _mupdf.FzTextSpan_items(self, i)

    def __init__(self, *args):
        """
        *Overload 1:*
        Default constructor, sets `m_internal` to null.

        |

        *Overload 2:*
        Constructor using raw copy of pre-existing `::fz_text_span`.
        """
        _mupdf.FzTextSpan_swiginit(self, _mupdf.new_FzTextSpan(*args))
    __swig_destroy__ = _mupdf.delete_FzTextSpan

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzTextSpan_m_internal_value(self)
    m_internal = property(_mupdf.FzTextSpan_m_internal_get, _mupdf.FzTextSpan_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzTextSpan_s_num_instances_get, _mupdf.FzTextSpan_s_num_instances_set)