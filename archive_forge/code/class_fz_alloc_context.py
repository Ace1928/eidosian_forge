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
class fz_alloc_context(object):
    """	Allocator structure; holds callbacks and private data pointer."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    user = property(_mupdf.fz_alloc_context_user_get, _mupdf.fz_alloc_context_user_set)
    malloc = property(_mupdf.fz_alloc_context_malloc_get, _mupdf.fz_alloc_context_malloc_set)
    realloc = property(_mupdf.fz_alloc_context_realloc_get, _mupdf.fz_alloc_context_realloc_set)
    free = property(_mupdf.fz_alloc_context_free_get, _mupdf.fz_alloc_context_free_set)

    def __init__(self):
        _mupdf.fz_alloc_context_swiginit(self, _mupdf.new_fz_alloc_context())
    __swig_destroy__ = _mupdf.delete_fz_alloc_context