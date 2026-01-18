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
class FzShade(object):
    """
    Wrapper class for struct `fz_shade`.
    Structure is public to allow derived classes. Do not
    access the members directly.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def fz_bound_shade(self, ctm):
        """
        Class-aware wrapper for `::fz_bound_shade()`.
        	Bound a given shading.

        	shade: The shade to bound.

        	ctm: The transform to apply to the shade before bounding.

        	r: Pointer to storage to put the bounds in.

        	Returns r, updated to contain the bounds for the shading.
        """
        return _mupdf.FzShade_fz_bound_shade(self, ctm)

    def fz_paint_shade(self, override_cs, ctm, dest, color_params, bbox, eop, cache):
        """
        Class-aware wrapper for `::fz_paint_shade()`.

        This method has out-params. Python/C# wrappers look like:
        	`fz_paint_shade(::fz_colorspace *override_cs, ::fz_matrix ctm, ::fz_pixmap *dest, ::fz_color_params color_params, ::fz_irect bbox, const ::fz_overprint *eop, ::fz_shade_color_cache **cache)` =>

        	Render a shade to a given pixmap.

        	shade: The shade to paint.

        	override_cs: NULL, or colorspace to override the shades
        	inbuilt colorspace.

        	ctm: The transform to apply.

        	dest: The pixmap to render into.

        	color_params: The color rendering settings

        	bbox: Pointer to a bounding box to limit the rendering
        	of the shade.

        	eop: NULL, or pointer to overprint bitmap.

        	cache: *cache is used to cache color information. If *cache is NULL it
        	is set to point to a new fz_shade_color_cache. If cache is NULL it is
        	ignored.
        """
        return _mupdf.FzShade_fz_paint_shade(self, override_cs, ctm, dest, color_params, bbox, eop, cache)

    def fz_process_shade(self, ctm, scissor, prepare, process, process_arg):
        """
        Class-aware wrapper for `::fz_process_shade()`.
        	Process a shade, using supplied callback functions. This
        	decomposes the shading to a mesh (even ones that are not
        	natively meshes, such as linear or radial shadings), and
        	processes triangles from those meshes.

        	shade: The shade to process.

        	ctm: The transform to use

        	prepare: Callback function to 'prepare' each vertex.
        	This function is passed an array of floats, and populates
        	a fz_vertex structure.

        	process: This function is passed 3 pointers to vertex
        	structures, and actually performs the processing (typically
        	filling the area between the vertexes).

        	process_arg: An opaque argument passed through from caller
        	to callback functions.
        """
        return _mupdf.FzShade_fz_process_shade(self, ctm, scissor, prepare, process, process_arg)

    def fz_paint_shade_no_cache(self, override_cs, ctm, dest, color_params, bbox, eop):
        """ Extra wrapper for fz_paint_shade(), passing cache=NULL."""
        return _mupdf.FzShade_fz_paint_shade_no_cache(self, override_cs, ctm, dest, color_params, bbox, eop)

    def __init__(self, *args):
        """
        *Overload 1:*
        Copy constructor using `fz_keep_shade()`.

        |

        *Overload 2:*
        Default constructor, sets `m_internal` to null.

        |

        *Overload 3:*
        Constructor using raw copy of pre-existing `::fz_shade`.
        """
        _mupdf.FzShade_swiginit(self, _mupdf.new_FzShade(*args))
    __swig_destroy__ = _mupdf.delete_FzShade

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzShade_m_internal_value(self)
    m_internal = property(_mupdf.FzShade_m_internal_get, _mupdf.FzShade_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzShade_s_num_instances_get, _mupdf.FzShade_s_num_instances_set)