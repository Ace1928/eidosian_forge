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
class FzColorspace(object):
    """
    Wrapper class for struct `fz_colorspace`.
    Describes a given colorspace.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    Fixed_GRAY = _mupdf.FzColorspace_Fixed_GRAY
    Fixed_RGB = _mupdf.FzColorspace_Fixed_RGB
    Fixed_BGR = _mupdf.FzColorspace_Fixed_BGR
    Fixed_CMYK = _mupdf.FzColorspace_Fixed_CMYK
    Fixed_LAB = _mupdf.FzColorspace_Fixed_LAB

    def fz_base_colorspace(self):
        """
        Class-aware wrapper for `::fz_base_colorspace()`.
        	Get the 'base' colorspace for a colorspace.

        	For indexed colorspaces, this is the colorspace the index
        	decodes into. For all other colorspaces, it is the colorspace
        	itself.

        	The returned colorspace is 'borrowed' (i.e. no additional
        	references are taken or dropped).
        """
        return _mupdf.FzColorspace_fz_base_colorspace(self)

    def fz_clamp_color(self, _in, out):
        """
        Class-aware wrapper for `::fz_clamp_color()`.

        This method has out-params. Python/C# wrappers look like:
        	`fz_clamp_color(const float *in)` => float out

        	Clamp the samples in a color to the correct ranges for a
        	given colorspace.
        """
        return _mupdf.FzColorspace_fz_clamp_color(self, _in, out)

    def fz_colorspace_colorant(self, n):
        """
        Class-aware wrapper for `::fz_colorspace_colorant()`.
        	Retrieve a the name for a colorant.

        	Returns a pointer with the same lifespan as the colorspace.
        """
        return _mupdf.FzColorspace_fz_colorspace_colorant(self, n)

    def fz_colorspace_device_n_has_cmyk(self):
        """
        Class-aware wrapper for `::fz_colorspace_device_n_has_cmyk()`.
        	True if DeviceN color space has cyan magenta yellow or black as
        	one of its colorants.
        """
        return _mupdf.FzColorspace_fz_colorspace_device_n_has_cmyk(self)

    def fz_colorspace_device_n_has_only_cmyk(self):
        """
        Class-aware wrapper for `::fz_colorspace_device_n_has_only_cmyk()`.
        	True if DeviceN color space has only colorants from the CMYK set.
        """
        return _mupdf.FzColorspace_fz_colorspace_device_n_has_only_cmyk(self)

    def fz_colorspace_is_cmyk(self):
        """ Class-aware wrapper for `::fz_colorspace_is_cmyk()`."""
        return _mupdf.FzColorspace_fz_colorspace_is_cmyk(self)

    def fz_colorspace_is_device(self):
        """ Class-aware wrapper for `::fz_colorspace_is_device()`."""
        return _mupdf.FzColorspace_fz_colorspace_is_device(self)

    def fz_colorspace_is_device_cmyk(self):
        """ Class-aware wrapper for `::fz_colorspace_is_device_cmyk()`."""
        return _mupdf.FzColorspace_fz_colorspace_is_device_cmyk(self)

    def fz_colorspace_is_device_gray(self):
        """ Class-aware wrapper for `::fz_colorspace_is_device_gray()`."""
        return _mupdf.FzColorspace_fz_colorspace_is_device_gray(self)

    def fz_colorspace_is_device_n(self):
        """ Class-aware wrapper for `::fz_colorspace_is_device_n()`."""
        return _mupdf.FzColorspace_fz_colorspace_is_device_n(self)

    def fz_colorspace_is_gray(self):
        """
        Class-aware wrapper for `::fz_colorspace_is_gray()`.
        	Tests for particular types of colorspaces
        """
        return _mupdf.FzColorspace_fz_colorspace_is_gray(self)

    def fz_colorspace_is_indexed(self):
        """ Class-aware wrapper for `::fz_colorspace_is_indexed()`."""
        return _mupdf.FzColorspace_fz_colorspace_is_indexed(self)

    def fz_colorspace_is_lab(self):
        """ Class-aware wrapper for `::fz_colorspace_is_lab()`."""
        return _mupdf.FzColorspace_fz_colorspace_is_lab(self)

    def fz_colorspace_is_lab_icc(self):
        """ Class-aware wrapper for `::fz_colorspace_is_lab_icc()`."""
        return _mupdf.FzColorspace_fz_colorspace_is_lab_icc(self)

    def fz_colorspace_is_rgb(self):
        """ Class-aware wrapper for `::fz_colorspace_is_rgb()`."""
        return _mupdf.FzColorspace_fz_colorspace_is_rgb(self)

    def fz_colorspace_is_subtractive(self):
        """
        Class-aware wrapper for `::fz_colorspace_is_subtractive()`.
        	True for CMYK, Separation and DeviceN colorspaces.
        """
        return _mupdf.FzColorspace_fz_colorspace_is_subtractive(self)

    def fz_colorspace_n(self):
        """
        Class-aware wrapper for `::fz_colorspace_n()`.
        	Query the number of colorants in a colorspace.
        """
        return _mupdf.FzColorspace_fz_colorspace_n(self)

    def fz_colorspace_name(self):
        """
        Class-aware wrapper for `::fz_colorspace_name()`.
        	Query the name of a colorspace.

        	The returned string has the same lifespan as the colorspace
        	does. Caller should not free it.
        """
        return _mupdf.FzColorspace_fz_colorspace_name(self)

    def fz_colorspace_name_colorant(self, n, name):
        """
        Class-aware wrapper for `::fz_colorspace_name_colorant()`.
        	Assign a name for a given colorant in a colorspace.

        	Used while initially setting up a colorspace. The string is
        	copied into local storage, so need not be retained by the
        	caller.
        """
        return _mupdf.FzColorspace_fz_colorspace_name_colorant(self, n, name)

    def fz_colorspace_type(self):
        """
        Class-aware wrapper for `::fz_colorspace_type()`.
        	Query the type of colorspace.
        """
        return _mupdf.FzColorspace_fz_colorspace_type(self)

    def fz_convert_color(self, sv, ds, dv, _is, params):
        """
        Class-aware wrapper for `::fz_convert_color()`.

        This method has out-params. Python/C# wrappers look like:
        	`fz_convert_color(const float *sv, ::fz_colorspace *ds, ::fz_colorspace *is, ::fz_color_params params)` => float dv

        	Convert color values sv from colorspace ss into colorvalues dv
        	for colorspace ds, via an optional intervening space is,
        	respecting the given color_params.
        """
        return _mupdf.FzColorspace_fz_convert_color(self, sv, ds, dv, _is, params)

    def fz_convert_separation_colors(self, src_color, dst_seps, dst_cs, dst_color, color_params):
        """
        Class-aware wrapper for `::fz_convert_separation_colors()`.

        This method has out-params. Python/C# wrappers look like:
        	`fz_convert_separation_colors(const float *src_color, ::fz_separations *dst_seps, ::fz_colorspace *dst_cs, ::fz_color_params color_params)` => float dst_color

        	Convert a color given in terms of one colorspace,
        	to a color in terms of another colorspace/separations.
        """
        return _mupdf.FzColorspace_fz_convert_separation_colors(self, src_color, dst_seps, dst_cs, dst_color, color_params)

    def fz_is_valid_blend_colorspace(self):
        """
        Class-aware wrapper for `::fz_is_valid_blend_colorspace()`.
        	Check to see that a colorspace is appropriate to be used as
        	a blending space (i.e. only grey, rgb or cmyk).
        """
        return _mupdf.FzColorspace_fz_is_valid_blend_colorspace(self)

    def fz_new_indexed_colorspace(self, high, lookup):
        """
        Class-aware wrapper for `::fz_new_indexed_colorspace()`.
        	Create an indexed colorspace.

        	The supplied lookup table is high palette entries long. Each
        	entry is n bytes long, where n is given by the number of
        	colorants in the base colorspace, one byte per colorant.

        	Ownership of lookup is passed it; it will be freed on
        	destruction, so must be heap allocated.

        	The colorspace will keep an additional reference to the base
        	colorspace that will be dropped on destruction.

        	The returned reference should be dropped when it is finished
        	with.

        	Colorspaces are immutable once created.
        """
        return _mupdf.FzColorspace_fz_new_indexed_colorspace(self, high, lookup)

    def fz_new_pixmap(self, w, h, seps, alpha):
        """
        Class-aware wrapper for `::fz_new_pixmap()`.
        	Create a new pixmap, with its origin at (0,0)

        	cs: The colorspace to use for the pixmap, or NULL for an alpha
        	plane/mask.

        	w: The width of the pixmap (in pixels)

        	h: The height of the pixmap (in pixels)

        	seps: Details of separations.

        	alpha: 0 for no alpha, 1 for alpha.

        	Returns a pointer to the new pixmap. Throws exception on failure
        	to allocate.
        """
        return _mupdf.FzColorspace_fz_new_pixmap(self, w, h, seps, alpha)

    def fz_new_pixmap_with_bbox(self, bbox, seps, alpha):
        """
        Class-aware wrapper for `::fz_new_pixmap_with_bbox()`.
        	Create a pixmap of a given size, location and pixel format.

        	The bounding box specifies the size of the created pixmap and
        	where it will be located. The colorspace determines the number
        	of components per pixel. Alpha is always present. Pixmaps are
        	reference counted, so drop references using fz_drop_pixmap.

        	colorspace: Colorspace format used for the created pixmap. The
        	pixmap will keep a reference to the colorspace.

        	bbox: Bounding box specifying location/size of created pixmap.

        	seps: Details of separations.

        	alpha: 0 for no alpha, 1 for alpha.

        	Returns a pointer to the new pixmap. Throws exception on failure
        	to allocate.
        """
        return _mupdf.FzColorspace_fz_new_pixmap_with_bbox(self, bbox, seps, alpha)

    def fz_new_pixmap_with_bbox_and_data(self, rect, seps, alpha, samples):
        """
        Class-aware wrapper for `::fz_new_pixmap_with_bbox_and_data()`.
        	Create a pixmap of a given size, location and pixel format,
        	using the supplied data block.

        	The bounding box specifies the size of the created pixmap and
        	where it will be located. The colorspace determines the number
        	of components per pixel. Alpha is always present. Pixmaps are
        	reference counted, so drop references using fz_drop_pixmap.

        	colorspace: Colorspace format used for the created pixmap. The
        	pixmap will keep a reference to the colorspace.

        	rect: Bounding box specifying location/size of created pixmap.

        	seps: Details of separations.

        	alpha: Number of alpha planes (0 or 1).

        	samples: The data block to keep the samples in.

        	Returns a pointer to the new pixmap. Throws exception on failure
        	to allocate.
        """
        return _mupdf.FzColorspace_fz_new_pixmap_with_bbox_and_data(self, rect, seps, alpha, samples)

    def fz_new_pixmap_with_data(self, w, h, seps, alpha, stride, samples):
        """
        Class-aware wrapper for `::fz_new_pixmap_with_data()`.
        	Create a new pixmap, with its origin at
        	(0,0) using the supplied data block.

        	cs: The colorspace to use for the pixmap, or NULL for an alpha
        	plane/mask.

        	w: The width of the pixmap (in pixels)

        	h: The height of the pixmap (in pixels)

        	seps: Details of separations.

        	alpha: 0 for no alpha, 1 for alpha.

        	stride: The byte offset from the pixel data in a row to the
        	pixel data in the next row.

        	samples: The data block to keep the samples in.

        	Returns a pointer to the new pixmap. Throws exception on failure to
        	allocate.
        """
        return _mupdf.FzColorspace_fz_new_pixmap_with_data(self, w, h, seps, alpha, stride, samples)

    def pdf_is_tint_colorspace(self):
        """ Class-aware wrapper for `::pdf_is_tint_colorspace()`."""
        return _mupdf.FzColorspace_pdf_is_tint_colorspace(self)

    def __init__(self, *args):
        """
        *Overload 1:*
         == Constructors.  Constructor using `fz_new_cal_gray_colorspace()`.
        		Create a calibrated gray colorspace.

        		The returned reference should be dropped when it is finished
        		with.

        		Colorspaces are immutable once created.


        |

        *Overload 2:*
         Constructor using `fz_new_cal_rgb_colorspace()`.
        		Create a calibrated rgb colorspace.

        		The returned reference should be dropped when it is finished
        		with.

        		Colorspaces are immutable once created.


        |

        *Overload 3:*
         Constructor using `fz_new_colorspace()`.
        		Creates a new colorspace instance and returns a reference.

        		No internal checking is done that the colorspace type (e.g.
        		CMYK) matches with the flags (e.g. FZ_COLORSPACE_HAS_CMYK) or
        		colorant count (n) or name.

        		The reference should be dropped when it is finished with.

        		Colorspaces are immutable once created (with the exception of
        		setting up colorant names for separation spaces).


        |

        *Overload 4:*
         Constructor using `fz_new_icc_colorspace()`.
        		Create a colorspace from an ICC profile supplied in buf.

        		Limited checking is done to ensure that the colorspace type is
        		appropriate for the supplied ICC profile.

        		An additional reference is taken to buf, which will be dropped
        		on destruction. Ownership is NOT passed in.

        		The returned reference should be dropped when it is finished
        		with.

        		Colorspaces are immutable once created.


        |

        *Overload 5:*
         Constructor using `fz_new_indexed_colorspace()`.
        		Create an indexed colorspace.

        		The supplied lookup table is high palette entries long. Each
        		entry is n bytes long, where n is given by the number of
        		colorants in the base colorspace, one byte per colorant.

        		Ownership of lookup is passed it; it will be freed on
        		destruction, so must be heap allocated.

        		The colorspace will keep an additional reference to the base
        		colorspace that will be dropped on destruction.

        		The returned reference should be dropped when it is finished
        		with.

        		Colorspaces are immutable once created.


        |

        *Overload 6:*
         Construct using one of: fz_device_gray(), fz_device_rgb(), fz_device_bgr(), fz_device_cmyk(), fz_device_lab().

        |

        *Overload 7:*
         Copy constructor using `fz_keep_colorspace()`.

        |

        *Overload 8:*
         Default constructor, sets `m_internal` to null.

        |

        *Overload 9:*
         Constructor using raw copy of pre-existing `::fz_colorspace`.
        """
        _mupdf.FzColorspace_swiginit(self, _mupdf.new_FzColorspace(*args))
    __swig_destroy__ = _mupdf.delete_FzColorspace

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzColorspace_m_internal_value(self)
    m_internal = property(_mupdf.FzColorspace_m_internal_get, _mupdf.FzColorspace_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzColorspace_s_num_instances_get, _mupdf.FzColorspace_s_num_instances_set)