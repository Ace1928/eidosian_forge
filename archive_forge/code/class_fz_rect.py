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
class fz_rect(object):
    """
    fz_rect is a rectangle represented by two diagonally opposite
    corners at arbitrary coordinates.

    Rectangles are always axis-aligned with the X- and Y- axes. We
    wish to distinguish rectangles in 3 categories; infinite, finite,
    and invalid. Zero area rectangles are a sub-category of finite
    ones.

    For all valid rectangles, x0 <= x1 and y0 <= y1 in all cases.
    Infinite rectangles have x0 = y0 = FZ_MIN_INF_RECT,
    x1 = y1 = FZ_MAX_INF_RECT. For any non infinite valid rectangle,
    the area is defined as (x1 - x0) * (y1 - y0).

    To check for empty or infinite rectangles use fz_is_empty_rect
    and fz_is_infinite_rect. To check for valid rectangles use
    fz_is_valid_rect.

    We choose this representation, so that we can easily distinguish
    the difference between intersecting 2 valid rectangles and
    getting an invalid one, as opposed to getting a zero area one
    (which nonetheless has valid bounds within the plane).

    x0, y0: The top left corner.

    x1, y1: The bottom right corner.

    We choose FZ_{MIN,MAX}_INF_RECT to be the largest 32bit signed
    integer values that survive roundtripping to floats.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    x0 = property(_mupdf.fz_rect_x0_get, _mupdf.fz_rect_x0_set)
    y0 = property(_mupdf.fz_rect_y0_get, _mupdf.fz_rect_y0_set)
    x1 = property(_mupdf.fz_rect_x1_get, _mupdf.fz_rect_x1_set)
    y1 = property(_mupdf.fz_rect_y1_get, _mupdf.fz_rect_y1_set)

    def __init__(self):
        _mupdf.fz_rect_swiginit(self, _mupdf.new_fz_rect())
    __swig_destroy__ = _mupdf.delete_fz_rect