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
class FzBitmap(object):
    """
    Wrapper class for struct `fz_bitmap`.
    Bitmaps have 1 bit per component. Only used for creating
    halftoned versions of contone buffers, and saving out. Samples
    are stored msb first, akin to pbms.

    The internals of this struct are considered implementation
    details and subject to change. Where possible, accessor
    functions should be used in preference.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def fz_bitmap_details(self, w, h, n, stride):
        """
        Class-aware wrapper for `::fz_bitmap_details()`.

        This method has out-params. Python/C# wrappers look like:
        	`fz_bitmap_details()` => `(int w, int h, int n, int stride)`

        	Retrieve details of a given bitmap.

        	bitmap: The bitmap to query.

        	w: Pointer to storage to retrieve width (or NULL).

        	h: Pointer to storage to retrieve height (or NULL).

        	n: Pointer to storage to retrieve number of color components (or
        	NULL).

        	stride: Pointer to storage to retrieve bitmap stride (or NULL).
        """
        return _mupdf.FzBitmap_fz_bitmap_details(self, w, h, n, stride)

    def fz_clear_bitmap(self):
        """
        Class-aware wrapper for `::fz_clear_bitmap()`.
        	Set the entire bitmap to 0.

        	Never throws exceptions.
        """
        return _mupdf.FzBitmap_fz_clear_bitmap(self)

    def fz_invert_bitmap(self):
        """
        Class-aware wrapper for `::fz_invert_bitmap()`.
        	Invert bitmap.

        	Never throws exceptions.
        """
        return _mupdf.FzBitmap_fz_invert_bitmap(self)

    def fz_save_bitmap_as_pbm(self, filename):
        """
        Class-aware wrapper for `::fz_save_bitmap_as_pbm()`.
        	Save a bitmap as a pbm.
        """
        return _mupdf.FzBitmap_fz_save_bitmap_as_pbm(self, filename)

    def fz_save_bitmap_as_pcl(self, filename, append, pcl):
        """
        Class-aware wrapper for `::fz_save_bitmap_as_pcl()`.
        	Save a bitmap as mono PCL.
        """
        return _mupdf.FzBitmap_fz_save_bitmap_as_pcl(self, filename, append, pcl)

    def fz_save_bitmap_as_pkm(self, filename):
        """
        Class-aware wrapper for `::fz_save_bitmap_as_pkm()`.
        	Save a CMYK bitmap as a pkm.
        """
        return _mupdf.FzBitmap_fz_save_bitmap_as_pkm(self, filename)

    def fz_save_bitmap_as_pwg(self, filename, append, pwg):
        """
        Class-aware wrapper for `::fz_save_bitmap_as_pwg()`.
        	Save a bitmap as a PWG.
        """
        return _mupdf.FzBitmap_fz_save_bitmap_as_pwg(self, filename, append, pwg)

    def __init__(self, *args):
        """
        *Overload 1:*
         == Constructors.  Constructor using `fz_new_bitmap()`.
        		Create a new bitmap.

        		w, h: Width and Height for the bitmap

        		n: Number of color components (assumed to be a divisor of 8)

        		xres, yres: X and Y resolutions (in pixels per inch).

        		Returns pointer to created bitmap structure. The bitmap
        		data is uninitialised.


        |

        *Overload 2:*
         Constructor using `fz_new_bitmap_from_pixmap()`.
        		Make a bitmap from a pixmap and a halftone.

        		pix: The pixmap to generate from. Currently must be a single
        		color component with no alpha.

        		ht: The halftone to use. NULL implies the default halftone.

        		Returns the resultant bitmap. Throws exceptions in the case of
        		failure to allocate.


        |

        *Overload 3:*
         Constructor using `fz_new_bitmap_from_pixmap_band()`.
        		Make a bitmap from a pixmap and a
        		halftone, allowing for the position of the pixmap within an
        		overall banded rendering.

        		pix: The pixmap to generate from. Currently must be a single
        		color component with no alpha.

        		ht: The halftone to use. NULL implies the default halftone.

        		band_start: Vertical offset within the overall banded rendering
        		(in pixels)

        		Returns the resultant bitmap. Throws exceptions in the case of
        		failure to allocate.


        |

        *Overload 4:*
         Copy constructor using `fz_keep_bitmap()`.

        |

        *Overload 5:*
         Default constructor, sets `m_internal` to null.

        |

        *Overload 6:*
         Constructor using raw copy of pre-existing `::fz_bitmap`.
        """
        _mupdf.FzBitmap_swiginit(self, _mupdf.new_FzBitmap(*args))

    def refs(self):
        return _mupdf.FzBitmap_refs(self)

    def w(self):
        return _mupdf.FzBitmap_w(self)

    def h(self):
        return _mupdf.FzBitmap_h(self)

    def stride(self):
        return _mupdf.FzBitmap_stride(self)

    def n(self):
        return _mupdf.FzBitmap_n(self)

    def xres(self):
        return _mupdf.FzBitmap_xres(self)

    def yres(self):
        return _mupdf.FzBitmap_yres(self)

    def samples(self):
        return _mupdf.FzBitmap_samples(self)
    __swig_destroy__ = _mupdf.delete_FzBitmap

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzBitmap_m_internal_value(self)
    m_internal = property(_mupdf.FzBitmap_m_internal_get, _mupdf.FzBitmap_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzBitmap_s_num_instances_get, _mupdf.FzBitmap_s_num_instances_set)