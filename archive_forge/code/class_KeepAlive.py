import ctypes
import io
import operator
import os
import sys
import weakref
from functools import reduce
from pathlib import Path
from tempfile import NamedTemporaryFile
from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontOptions, _encode_string
class KeepAlive(object):
    """
    Keep some objects alive until a callback is called.
    :attr:`closure` is a tuple of cairo_destroy_func_t and void* cdata objects,
    as expected by cairo_surface_set_mime_data().

    Either :meth:`save` must be called before the callback,
    or none of them must be called.

    """
    instances = set()

    def __init__(self, *objects):
        self.objects = objects
        weakself = weakref.ref(self)

        def closure(_):
            value = weakself()
            if value is not None:
                value.instances.remove(value)
        callback = ffi.callback('cairo_destroy_func_t', closure)
        self.closure = (callback, callback)

    def save(self):
        """Start keeping a reference to the passed objects."""
        self.instances.add(self)