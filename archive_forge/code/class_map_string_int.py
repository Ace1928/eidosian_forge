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
class map_string_int(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def iterator(self):
        return _mupdf.map_string_int_iterator(self)

    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _mupdf.map_string_int___nonzero__(self)

    def __bool__(self):
        return _mupdf.map_string_int___bool__(self)

    def __len__(self):
        return _mupdf.map_string_int___len__(self)

    def __iter__(self):
        return self.key_iterator()

    def iterkeys(self):
        return self.key_iterator()

    def itervalues(self):
        return self.value_iterator()

    def iteritems(self):
        return self.iterator()

    def __getitem__(self, key):
        return _mupdf.map_string_int___getitem__(self, key)

    def __delitem__(self, key):
        return _mupdf.map_string_int___delitem__(self, key)

    def has_key(self, key):
        return _mupdf.map_string_int_has_key(self, key)

    def keys(self):
        return _mupdf.map_string_int_keys(self)

    def values(self):
        return _mupdf.map_string_int_values(self)

    def items(self):
        return _mupdf.map_string_int_items(self)

    def __contains__(self, key):
        return _mupdf.map_string_int___contains__(self, key)

    def key_iterator(self):
        return _mupdf.map_string_int_key_iterator(self)

    def value_iterator(self):
        return _mupdf.map_string_int_value_iterator(self)

    def __setitem__(self, *args):
        return _mupdf.map_string_int___setitem__(self, *args)

    def asdict(self):
        return _mupdf.map_string_int_asdict(self)

    def __init__(self, *args):
        _mupdf.map_string_int_swiginit(self, _mupdf.new_map_string_int(*args))

    def empty(self):
        return _mupdf.map_string_int_empty(self)

    def size(self):
        return _mupdf.map_string_int_size(self)

    def swap(self, v):
        return _mupdf.map_string_int_swap(self, v)

    def begin(self):
        return _mupdf.map_string_int_begin(self)

    def end(self):
        return _mupdf.map_string_int_end(self)

    def rbegin(self):
        return _mupdf.map_string_int_rbegin(self)

    def rend(self):
        return _mupdf.map_string_int_rend(self)

    def clear(self):
        return _mupdf.map_string_int_clear(self)

    def get_allocator(self):
        return _mupdf.map_string_int_get_allocator(self)

    def count(self, x):
        return _mupdf.map_string_int_count(self, x)

    def erase(self, *args):
        return _mupdf.map_string_int_erase(self, *args)

    def find(self, x):
        return _mupdf.map_string_int_find(self, x)

    def lower_bound(self, x):
        return _mupdf.map_string_int_lower_bound(self, x)

    def upper_bound(self, x):
        return _mupdf.map_string_int_upper_bound(self, x)
    __swig_destroy__ = _mupdf.delete_map_string_int