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
class FzPath(object):
    """
     Wrapper class for struct `fz_path`.
    Vector path buffer.
    It can be stroked and dashed, or be filled.
    It has a fill rule (nonzero or even_odd).

    When rendering, they are flattened, stroked and dashed straight
    into the Global Edge List.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def fz_bound_path(self, stroke, ctm):
        """
        Class-aware wrapper for `::fz_bound_path()`.
        	Return a bounding rectangle for a path.

        	path: The path to bound.

        	stroke: If NULL, the bounding rectangle given is for
        	the filled path. If non-NULL the bounding rectangle
        	given is for the path stroked with the given attributes.

        	ctm: The matrix to apply to the path during stroking.

        	r: Pointer to a fz_rect which will be used to hold
        	the result.

        	Returns r, updated to contain the bounding rectangle.
        """
        return _mupdf.FzPath_fz_bound_path(self, stroke, ctm)

    def fz_clone_path(self):
        """
        Class-aware wrapper for `::fz_clone_path()`.
        	Clone the data for a path.

        	This is used in preference to fz_keep_path when a whole
        	new copy of a path is required, rather than just a shared
        	pointer. This probably indicates that the path is about to
        	be modified.

        	path: path to clone.

        	Throws exceptions on failure to allocate.
        """
        return _mupdf.FzPath_fz_clone_path(self)

    def fz_closepath(self):
        """
        Class-aware wrapper for `::fz_closepath()`.
        	Close the current subpath.

        	path: The path to modify.

        	Throws exceptions on failure to allocate, attempting to modify
        	a packed path, and illegal path closes (i.e. closing a non open
        	path).
        """
        return _mupdf.FzPath_fz_closepath(self)

    def fz_currentpoint(self):
        """
        Class-aware wrapper for `::fz_currentpoint()`.
        	Return the current point that a path has
        	reached or (0,0) if empty.

        	path: path to return the current point of.
        """
        return _mupdf.FzPath_fz_currentpoint(self)

    def fz_curveto(self, x0, y0, x1, y1, x2, y2):
        """
        Class-aware wrapper for `::fz_curveto()`.
        	Append a 'curveto' command to an open path. (For a
        	cubic bezier).

        	path: The path to modify.

        	x0, y0: The coordinates of the first control point for the
        	curve.

        	x1, y1: The coordinates of the second control point for the
        	curve.

        	x2, y2: The end coordinates for the curve.

        	Throws exceptions on failure to allocate, or attempting to
        	modify a packed path.
        """
        return _mupdf.FzPath_fz_curveto(self, x0, y0, x1, y1, x2, y2)

    def fz_curvetov(self, x1, y1, x2, y2):
        """
        Class-aware wrapper for `::fz_curvetov()`.
        	Append a 'curvetov' command to an open path. (For a
        	cubic bezier with the first control coordinate equal to
        	the start point).

        	path: The path to modify.

        	x1, y1: The coordinates of the second control point for the
        	curve.

        	x2, y2: The end coordinates for the curve.

        	Throws exceptions on failure to allocate, or attempting to
        	modify a packed path.
        """
        return _mupdf.FzPath_fz_curvetov(self, x1, y1, x2, y2)

    def fz_curvetoy(self, x0, y0, x2, y2):
        """
        Class-aware wrapper for `::fz_curvetoy()`.
        	Append a 'curvetoy' command to an open path. (For a
        	cubic bezier with the second control coordinate equal to
        	the end point).

        	path: The path to modify.

        	x0, y0: The coordinates of the first control point for the
        	curve.

        	x2, y2: The end coordinates for the curve (and the second
        	control coordinate).

        	Throws exceptions on failure to allocate, or attempting to
        	modify a packed path.
        """
        return _mupdf.FzPath_fz_curvetoy(self, x0, y0, x2, y2)

    def fz_lineto(self, x, y):
        """
        Class-aware wrapper for `::fz_lineto()`.
        	Append a 'lineto' command to an open path.

        	path: The path to modify.

        	x, y: The coordinate to line to.

        	Throws exceptions on failure to allocate, or attempting to
        	modify a packed path.
        """
        return _mupdf.FzPath_fz_lineto(self, x, y)

    def fz_moveto(self, x, y):
        """
        Class-aware wrapper for `::fz_moveto()`.
        	Append a 'moveto' command to a path.
        	This 'opens' a path.

        	path: The path to modify.

        	x, y: The coordinate to move to.

        	Throws exceptions on failure to allocate, or attempting to
        	modify a packed path.
        """
        return _mupdf.FzPath_fz_moveto(self, x, y)

    def fz_packed_path_size(self):
        """
        Class-aware wrapper for `::fz_packed_path_size()`.
        	Return the number of bytes required to pack a path.
        """
        return _mupdf.FzPath_fz_packed_path_size(self)

    def fz_quadto(self, x0, y0, x1, y1):
        """
        Class-aware wrapper for `::fz_quadto()`.
        	Append a 'quadto' command to an open path. (For a
        	quadratic bezier).

        	path: The path to modify.

        	x0, y0: The control coordinates for the quadratic curve.

        	x1, y1: The end coordinates for the quadratic curve.

        	Throws exceptions on failure to allocate, or attempting to
        	modify a packed path.
        """
        return _mupdf.FzPath_fz_quadto(self, x0, y0, x1, y1)

    def fz_rectto(self, x0, y0, x1, y1):
        """
        Class-aware wrapper for `::fz_rectto()`.
        	Append a 'rectto' command to an open path.

        	The rectangle is equivalent to:
        		moveto x0 y0
        		lineto x1 y0
        		lineto x1 y1
        		lineto x0 y1
        		closepath

        	path: The path to modify.

        	x0, y0: First corner of the rectangle.

        	x1, y1: Second corner of the rectangle.

        	Throws exceptions on failure to allocate, or attempting to
        	modify a packed path.
        """
        return _mupdf.FzPath_fz_rectto(self, x0, y0, x1, y1)

    def fz_transform_path(self, transform):
        """
        Class-aware wrapper for `::fz_transform_path()`.
        	Transform a path by a given
        	matrix.

        	path: The path to modify (must not be a packed path).

        	transform: The transform to apply.

        	Throws exceptions if the path is packed, or on failure
        	to allocate.
        """
        return _mupdf.FzPath_fz_transform_path(self, transform)

    def fz_trim_path(self):
        """
        Class-aware wrapper for `::fz_trim_path()`.
        	Minimise the internal storage used by a path.

        	As paths are constructed, the internal buffers
        	grow. To avoid repeated reallocations they
        	grow with some spare space. Once a path has
        	been fully constructed, this call allows the
        	excess space to be trimmed.
        """
        return _mupdf.FzPath_fz_trim_path(self)

    def fz_walk_path(self, walker, arg):
        """
        Class-aware wrapper for `::fz_walk_path()`.
        	Walk the segments of a path, calling the
        	appropriate callback function from a given set for each
        	segment of the path.

        	path: The path to walk.

        	walker: The set of callback functions to use. The first
        	4 callback pointers in the set must be non-NULL. The
        	subsequent ones can either be supplied, or can be left
        	as NULL, in which case the top 4 functions will be
        	called as appropriate to simulate them.

        	arg: An opaque argument passed in to each callback.

        	Exceptions will only be thrown if the underlying callback
        	functions throw them.
        """
        return _mupdf.FzPath_fz_walk_path(self, walker, arg)

    def __init__(self, *args):
        """
        *Overload 1:*
         == Constructors.  Constructor using `fz_new_path()`.
        		Create a new (empty) path structure.


        |

        *Overload 2:*
         Copy constructor using `fz_keep_path()`.

        |

        *Overload 3:*
         Constructor using raw copy of pre-existing `::fz_path`.
        """
        _mupdf.FzPath_swiginit(self, _mupdf.new_FzPath(*args))
    __swig_destroy__ = _mupdf.delete_FzPath

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzPath_m_internal_value(self)
    m_internal = property(_mupdf.FzPath_m_internal_get, _mupdf.FzPath_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzPath_s_num_instances_get, _mupdf.FzPath_s_num_instances_set)