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
class fz_key_storable(object):
    """
    Any storable object that can appear in the key of another
    storable object should include an fz_key_storable structure
    at the start (by convention at least) of their structure.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    storable = property(_mupdf.fz_key_storable_storable_get, _mupdf.fz_key_storable_storable_set)
    store_key_refs = property(_mupdf.fz_key_storable_store_key_refs_get, _mupdf.fz_key_storable_store_key_refs_set)

    def __init__(self):
        _mupdf.fz_key_storable_swiginit(self, _mupdf.new_fz_key_storable())
    __swig_destroy__ = _mupdf.delete_fz_key_storable