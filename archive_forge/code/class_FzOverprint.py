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
class FzOverprint(object):
    """
    Wrapper class for struct `fz_overprint`. Not copyable or assignable.
    Pixmaps represent a set of pixels for a 2 dimensional region of
    a plane. Each pixel has n components per pixel. The components
    are in the order process-components, spot-colors, alpha, where
    there can be 0 of any of those types. The data is in
    premultiplied alpha when rendering, but non-premultiplied for
    colorspace conversions and rescaling.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def __init__(self, *args):
        """
        *Overload 1:*
        Default constructor, sets `m_internal` to null.

        |

        *Overload 2:*
        Constructor using raw copy of pre-existing `::fz_overprint`.
        """
        _mupdf.FzOverprint_swiginit(self, _mupdf.new_FzOverprint(*args))
    __swig_destroy__ = _mupdf.delete_FzOverprint

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzOverprint_m_internal_value(self)
    m_internal = property(_mupdf.FzOverprint_m_internal_get, _mupdf.FzOverprint_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzOverprint_s_num_instances_get, _mupdf.FzOverprint_s_num_instances_set)