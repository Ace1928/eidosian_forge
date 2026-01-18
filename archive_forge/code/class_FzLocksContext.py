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
class FzLocksContext(object):
    """
    Wrapper class for struct `fz_locks_context`. Not copyable or assignable.
    Locking functions

    MuPDF is kept deliberately free of any knowledge of particular
    threading systems. As such, in order for safe multi-threaded
    operation, we rely on callbacks to client provided functions.

    A client is expected to provide FZ_LOCK_MAX number of mutexes,
    and a function to lock/unlock each of them. These may be
    recursive mutexes, but do not have to be.

    If a client does not intend to use multiple threads, then it
    may pass NULL instead of a lock structure.

    In order to avoid deadlocks, we have one simple rule
    internally as to how we use locks: We can never take lock n
    when we already hold any lock i, where 0 <= i <= n. In order
    to verify this, we have some debugging code, that can be
    enabled by defining FITZ_DEBUG_LOCKING.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def __init__(self, *args):
        """
        *Overload 1:*
        Default constructor, sets `m_internal` to null.

        |

        *Overload 2:*
        Constructor using raw copy of pre-existing `::fz_locks_context`.
        """
        _mupdf.FzLocksContext_swiginit(self, _mupdf.new_FzLocksContext(*args))
    __swig_destroy__ = _mupdf.delete_FzLocksContext

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzLocksContext_m_internal_value(self)
    m_internal = property(_mupdf.FzLocksContext_m_internal_get, _mupdf.FzLocksContext_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzLocksContext_s_num_instances_get, _mupdf.FzLocksContext_s_num_instances_set)