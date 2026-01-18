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
class FzInt2Heap(object):
    """ Wrapper class for struct `fz_int2_heap`. Not copyable or assignable."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def fz_int2_heap_insert(self, v):
        """ Class-aware wrapper for `::fz_int2_heap_insert()`."""
        return _mupdf.FzInt2Heap_fz_int2_heap_insert(self, v)

    def fz_int2_heap_sort(self):
        """ Class-aware wrapper for `::fz_int2_heap_sort()`."""
        return _mupdf.FzInt2Heap_fz_int2_heap_sort(self)

    def fz_int2_heap_uniq(self):
        """ Class-aware wrapper for `::fz_int2_heap_uniq()`."""
        return _mupdf.FzInt2Heap_fz_int2_heap_uniq(self)

    def __init__(self, *args):
        """
        *Overload 1:*
        Default constructor, sets `m_internal` to null.

        |

        *Overload 2:*
        Constructor using raw copy of pre-existing `::fz_int2_heap`.
        """
        _mupdf.FzInt2Heap_swiginit(self, _mupdf.new_FzInt2Heap(*args))
    __swig_destroy__ = _mupdf.delete_FzInt2Heap

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzInt2Heap_m_internal_value(self)
    m_internal = property(_mupdf.FzInt2Heap_m_internal_get, _mupdf.FzInt2Heap_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzInt2Heap_s_num_instances_get, _mupdf.FzInt2Heap_s_num_instances_set)