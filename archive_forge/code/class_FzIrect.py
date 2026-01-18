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
class FzIrect(object):
    """
    Wrapper class for struct `fz_irect`.
    fz_irect is a rectangle using integers instead of floats.

    It's used in the draw device and for pixmap dimensions.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def fz_expand_irect(self, expand):
        """ Class-aware wrapper for `::fz_expand_irect()`."""
        return _mupdf.FzIrect_fz_expand_irect(self, expand)

    def fz_intersect_irect(self, b):
        """
        Class-aware wrapper for `::fz_intersect_irect()`.
        	Compute intersection of two bounding boxes.

        	Similar to fz_intersect_rect but operates on two bounding
        	boxes instead of two rectangles.
        """
        return _mupdf.FzIrect_fz_intersect_irect(self, b)

    def fz_irect_height(self):
        """
        Class-aware wrapper for `::fz_irect_height()`.
        	Return the height of an irect. Invalid irects return 0.
        """
        return _mupdf.FzIrect_fz_irect_height(self)

    def fz_irect_width(self):
        """
        Class-aware wrapper for `::fz_irect_width()`.
        	Return the width of an irect. Invalid irects return 0.
        """
        return _mupdf.FzIrect_fz_irect_width(self)

    def fz_is_empty_irect(self):
        """ Class-aware wrapper for `::fz_is_empty_irect()`."""
        return _mupdf.FzIrect_fz_is_empty_irect(self)

    def fz_is_infinite_irect(self):
        """
        Class-aware wrapper for `::fz_is_infinite_irect()`.
        	Check if an integer rectangle
        	is infinite.
        """
        return _mupdf.FzIrect_fz_is_infinite_irect(self)

    def fz_is_valid_irect(self):
        """
        Class-aware wrapper for `::fz_is_valid_irect()`.
        	Check if an integer rectangle is valid.
        """
        return _mupdf.FzIrect_fz_is_valid_irect(self)

    def fz_rect_from_irect(self):
        """
        Class-aware wrapper for `::fz_rect_from_irect()`.
        	Convert a bbox into a rect.

        	For our purposes, a rect can represent all the values we meet in
        	a bbox, so nothing can go wrong.

        	rect: A place to store the generated rectangle.

        	bbox: The bbox to convert.

        	Returns rect (updated).
        """
        return _mupdf.FzIrect_fz_rect_from_irect(self)

    def fz_translate_irect(self, xoff, yoff):
        """ Class-aware wrapper for `::fz_translate_irect()`."""
        return _mupdf.FzIrect_fz_translate_irect(self, xoff, yoff)

    def __init__(self, *args):
        """
        *Overload 1:*
         == Constructors.  Constructor using `fz_irect_from_rect()`.
        		Convert a rect into the minimal bounding box
        		that covers the rectangle.

        		Coordinates in a bounding box are integers, so rounding of the
        		rects coordinates takes place. The top left corner is rounded
        		upwards and left while the bottom right corner is rounded
        		downwards and to the right.


        |

        *Overload 2:*
         Constructor using `fz_make_irect()`.

        |

        *Overload 3:*
         We use default copy constructor and operator=.  Default constructor, sets each member to default value.

        |

        *Overload 4:*
         Constructor using raw copy of pre-existing `::fz_irect`.

        |

        *Overload 5:*
         Constructor using raw copy of pre-existing `::fz_irect`.
        """
        _mupdf.FzIrect_swiginit(self, _mupdf.new_FzIrect(*args))

    def internal(self, *args):
        """
        *Overload 1:*
        Access as underlying struct.

        |

        *Overload 2:*
        Access as underlying struct.
        """
        return _mupdf.FzIrect_internal(self, *args)
    __swig_destroy__ = _mupdf.delete_FzIrect
    x0 = property(_mupdf.FzIrect_x0_get, _mupdf.FzIrect_x0_set)
    y0 = property(_mupdf.FzIrect_y0_get, _mupdf.FzIrect_y0_set)
    x1 = property(_mupdf.FzIrect_x1_get, _mupdf.FzIrect_x1_set)
    y1 = property(_mupdf.FzIrect_y1_get, _mupdf.FzIrect_y1_set)
    s_num_instances = property(_mupdf.FzIrect_s_num_instances_get, _mupdf.FzIrect_s_num_instances_set)

    def to_string(self):
        """ Returns string containing our members, labelled and inside (...), using operator<<."""
        return _mupdf.FzIrect_to_string(self)

    def __eq__(self, rhs):
        """ Comparison method."""
        return _mupdf.FzIrect___eq__(self, rhs)

    def __ne__(self, rhs):
        """ Comparison method."""
        return _mupdf.FzIrect___ne__(self, rhs)