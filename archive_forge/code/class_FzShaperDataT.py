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
class FzShaperDataT(object):
    """
    Wrapper class for struct `fz_shaper_data_t`. Not copyable or assignable.
    In order to shape a given font, we need to
    declare it to a shaper library (harfbuzz, by default, but others
    are possible). To avoid redeclaring it every time we need to
    shape, we hold a shaper handle and the destructor for it within
    the font itself. The handle is initialised by the caller when
    first required and the destructor is called when the fz_font is
    destroyed.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def __init__(self, *args):
        """
        *Overload 1:*
        Default constructor, sets `m_internal` to null.

        |

        *Overload 2:*
        Constructor using raw copy of pre-existing `::fz_shaper_data_t`.
        """
        _mupdf.FzShaperDataT_swiginit(self, _mupdf.new_FzShaperDataT(*args))
    __swig_destroy__ = _mupdf.delete_FzShaperDataT

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzShaperDataT_m_internal_value(self)
    m_internal = property(_mupdf.FzShaperDataT_m_internal_get, _mupdf.FzShaperDataT_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzShaperDataT_s_num_instances_get, _mupdf.FzShaperDataT_s_num_instances_set)