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
class fz_outline_iterator(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    drop = property(_mupdf.fz_outline_iterator_drop_get, _mupdf.fz_outline_iterator_drop_set)
    item = property(_mupdf.fz_outline_iterator_item_get, _mupdf.fz_outline_iterator_item_set)
    next = property(_mupdf.fz_outline_iterator_next_get, _mupdf.fz_outline_iterator_next_set)
    prev = property(_mupdf.fz_outline_iterator_prev_get, _mupdf.fz_outline_iterator_prev_set)
    up = property(_mupdf.fz_outline_iterator_up_get, _mupdf.fz_outline_iterator_up_set)
    down = property(_mupdf.fz_outline_iterator_down_get, _mupdf.fz_outline_iterator_down_set)
    insert = property(_mupdf.fz_outline_iterator_insert_get, _mupdf.fz_outline_iterator_insert_set)
    update = property(_mupdf.fz_outline_iterator_update_get, _mupdf.fz_outline_iterator_update_set)
    _del = property(_mupdf.fz_outline_iterator__del_get, _mupdf.fz_outline_iterator__del_set)
    doc = property(_mupdf.fz_outline_iterator_doc_get, _mupdf.fz_outline_iterator_doc_set)

    def __init__(self):
        _mupdf.fz_outline_iterator_swiginit(self, _mupdf.new_fz_outline_iterator())
    __swig_destroy__ = _mupdf.delete_fz_outline_iterator