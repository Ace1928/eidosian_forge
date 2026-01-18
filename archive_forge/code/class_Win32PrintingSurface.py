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
class Win32PrintingSurface(Surface):
    """Creates a cairo surface that targets the given DC.

    The DC will be queried for its initial clip extents,
    and this will be used as the size of the cairo surface.
    The DC should be a printing DC; antialiasing will be ignored,
    and GDI will be used as much as possible to draw to the surface.

    The returned surface will be wrapped using the paginated surface
    to provide correct complex rendering behaviour;
    cairo_surface_show_page() and associated methods must be used
    for correct output.

    :param hdc:
        The DC to create a surface for,
        as obtained from ``win32gui.CreateDC``.
        **Note**: this unsafely inteprets an integer as a pointer.
        Make sure it actually points to a valid DC!
    :type hdc: int

    *New in cairocffi 0.6*

    """

    def __init__(self, hdc):
        pointer = cairo.cairo_win32_printing_surface_create(ffi.cast('void*', hdc))
        Surface.__init__(self, pointer)