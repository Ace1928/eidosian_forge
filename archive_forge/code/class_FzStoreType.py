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
class FzStoreType(object):
    """
    Wrapper class for struct `fz_store_type`. Not copyable or assignable.
    Every type of object to be placed into the store defines an
    fz_store_type. This contains the pointers to functions to
    make hashes, manipulate keys, and check for needing reaping.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def __init__(self, *args):
        """
        *Overload 1:*
        Default constructor, sets `m_internal` to null.

        |

        *Overload 2:*
        Constructor using raw copy of pre-existing `::fz_store_type`.
        """
        _mupdf.FzStoreType_swiginit(self, _mupdf.new_FzStoreType(*args))
    __swig_destroy__ = _mupdf.delete_FzStoreType

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzStoreType_m_internal_value(self)
    m_internal = property(_mupdf.FzStoreType_m_internal_get, _mupdf.FzStoreType_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzStoreType_s_num_instances_get, _mupdf.FzStoreType_s_num_instances_set)