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
class FzPoint(object):
    """
    Wrapper class for struct `fz_point`.
    fz_point is a point in a two-dimensional space.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    @staticmethod
    def fz_transform_point_xy(x, y, m):
        """ Class-aware wrapper for `::fz_transform_point_xy()`."""
        return _mupdf.FzPoint_fz_transform_point_xy(x, y, m)

    def fz_is_point_inside_quad(self, q):
        """
        Class-aware wrapper for `::fz_is_point_inside_quad()`.
        	Inclusion test for quads.
        """
        return _mupdf.FzPoint_fz_is_point_inside_quad(self, q)

    def fz_is_point_inside_rect(self, r):
        """
        Class-aware wrapper for `::fz_is_point_inside_rect()`.
        	Inclusion test for rects. (Rect is assumed to be open, i.e.
        	top right corner is not included).
        """
        return _mupdf.FzPoint_fz_is_point_inside_rect(self, r)

    def fz_normalize_vector(self):
        """
        Class-aware wrapper for `::fz_normalize_vector()`.
        	Normalize a vector to length one.
        """
        return _mupdf.FzPoint_fz_normalize_vector(self)

    def fz_transform_point(self, *args):
        """
        *Overload 1:*
         Class-aware wrapper for `::fz_transform_point()`.
        		Apply a transformation to a point.

        		transform: Transformation matrix to apply. See fz_concat,
        		fz_scale, fz_rotate and fz_translate for how to create a
        		matrix.

        		point: Pointer to point to update.

        		Returns transform (unchanged).


        |

        *Overload 2:*
         Class-aware wrapper for `::fz_transform_point()`.
        		Apply a transformation to a point.

        		transform: Transformation matrix to apply. See fz_concat,
        		fz_scale, fz_rotate and fz_translate for how to create a
        		matrix.

        		point: Pointer to point to update.

        		Returns transform (unchanged).
        """
        return _mupdf.FzPoint_fz_transform_point(self, *args)

    def fz_transform_vector(self, *args):
        """
        *Overload 1:*
         Class-aware wrapper for `::fz_transform_vector()`.
        		Apply a transformation to a vector.

        		transform: Transformation matrix to apply. See fz_concat,
        		fz_scale and fz_rotate for how to create a matrix. Any
        		translation will be ignored.

        		vector: Pointer to vector to update.


        |

        *Overload 2:*
         Class-aware wrapper for `::fz_transform_vector()`.
        		Apply a transformation to a vector.

        		transform: Transformation matrix to apply. See fz_concat,
        		fz_scale and fz_rotate for how to create a matrix. Any
        		translation will be ignored.

        		vector: Pointer to vector to update.
        """
        return _mupdf.FzPoint_fz_transform_vector(self, *args)

    def transform(self, m):
        """ Post-multiply *this by <m> and return *this."""
        return _mupdf.FzPoint_transform(self, m)

    def __init__(self, *args):
        """
        *Overload 1:*
        Construct using specified values.

        |

        *Overload 2:*
        We use default copy constructor and operator=.  Default constructor, sets each member to default value.

        |

        *Overload 3:*
        Constructor using raw copy of pre-existing `::fz_point`.

        |

        *Overload 4:*
        Constructor using raw copy of pre-existing `::fz_point`.
        """
        _mupdf.FzPoint_swiginit(self, _mupdf.new_FzPoint(*args))

    def internal(self, *args):
        """
        *Overload 1:*
        Access as underlying struct.

        |

        *Overload 2:*
        Access as underlying struct.
        """
        return _mupdf.FzPoint_internal(self, *args)
    __swig_destroy__ = _mupdf.delete_FzPoint
    x = property(_mupdf.FzPoint_x_get, _mupdf.FzPoint_x_set)
    y = property(_mupdf.FzPoint_y_get, _mupdf.FzPoint_y_set)
    s_num_instances = property(_mupdf.FzPoint_s_num_instances_get, _mupdf.FzPoint_s_num_instances_set)

    def to_string(self):
        """ Returns string containing our members, labelled and inside (...), using operator<<."""
        return _mupdf.FzPoint_to_string(self)

    def __eq__(self, rhs):
        """ Comparison method."""
        return _mupdf.FzPoint___eq__(self, rhs)

    def __ne__(self, rhs):
        """ Comparison method."""
        return _mupdf.FzPoint___ne__(self, rhs)