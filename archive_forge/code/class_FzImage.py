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
class FzImage(object):
    """
    Wrapper class for struct `fz_image`.
    Images are storable objects from which we can obtain fz_pixmaps.
    These may be implemented as simple wrappers around a pixmap, or
    as more complex things that decode at different subsample
    settings on demand.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def fz_compressed_image_buffer(self):
        """
        Class-aware wrapper for `::fz_compressed_image_buffer()`.
        	Retrieve the underlying compressed data for an image.

        	Returns a pointer to the underlying data buffer for an image,
        	or NULL if this image is not based upon a compressed data
        	buffer.

        	This is not a reference counted structure, so no reference is
        	returned. Lifespan is limited to that of the image itself.
        """
        return _mupdf.FzImage_fz_compressed_image_buffer(self)

    def fz_compressed_image_type(self):
        """
        Class-aware wrapper for `::fz_compressed_image_type()`.
        	Return the type of a compressed image.

        	Any non-compressed image will have the type returned as UNKNOWN.
        """
        return _mupdf.FzImage_fz_compressed_image_type(self)

    def fz_get_pixmap_from_image(self, subarea, ctm, w, h):
        """
        Class-aware wrapper for `::fz_get_pixmap_from_image()`.

        This method has out-params. Python/C# wrappers look like:
        	`fz_get_pixmap_from_image(const ::fz_irect *subarea, ::fz_matrix *ctm)` => `(fz_pixmap *, int w, int h)`

        	Called to get a handle to a pixmap from an image.

        	image: The image to retrieve a pixmap from.

        	subarea: The subarea of the image that we actually care about
        	(or NULL to indicate the whole image).

        	ctm: Optional, unless subarea is given. If given, then on
        	entry this is the transform that will be applied to the complete
        	image. It should be updated on exit to the transform to apply to
        	the given subarea of the image. This is used to calculate the
        	desired width/height for subsampling.

        	w: If non-NULL, a pointer to an int to be updated on exit to the
        	width (in pixels) that the scaled output will cover.

        	h: If non-NULL, a pointer to an int to be updated on exit to the
        	height (in pixels) that the scaled output will cover.

        	Returns a non NULL kept pixmap pointer. May throw exceptions.
        """
        return _mupdf.FzImage_fz_get_pixmap_from_image(self, subarea, ctm, w, h)

    def fz_get_unscaled_pixmap_from_image(self):
        """
        Class-aware wrapper for `::fz_get_unscaled_pixmap_from_image()`.
        	Calls fz_get_pixmap_from_image() with ctm, subarea, w and h all set to NULL.
        """
        return _mupdf.FzImage_fz_get_unscaled_pixmap_from_image(self)

    def fz_image_orientation(self):
        """
        Class-aware wrapper for `::fz_image_orientation()`.
        	Request the natural orientation of an image.

        	This is for images (such as JPEG) that can contain internal
        	specifications of rotation/flips. This is ignored by all the
        	internal decode/rendering routines, but can be used by callers
        	(such as the image document handler) to respect such
        	specifications.

        	The values used by MuPDF are as follows, with the equivalent
        	Exif specifications given for information:

        	0: Undefined
        	1: 0 degree ccw rotation. (Exif = 1)
        	2: 90 degree ccw rotation. (Exif = 8)
        	3: 180 degree ccw rotation. (Exif = 3)
        	4: 270 degree ccw rotation. (Exif = 6)
        	5: flip on X. (Exif = 2)
        	6: flip on X, then rotate ccw by 90 degrees. (Exif = 5)
        	7: flip on X, then rotate ccw by 180 degrees. (Exif = 4)
        	8: flip on X, then rotate ccw by 270 degrees. (Exif = 7)
        """
        return _mupdf.FzImage_fz_image_orientation(self)

    def fz_image_orientation_matrix(self):
        """ Class-aware wrapper for `::fz_image_orientation_matrix()`."""
        return _mupdf.FzImage_fz_image_orientation_matrix(self)

    def fz_image_resolution(self, xres, yres):
        """
        Class-aware wrapper for `::fz_image_resolution()`.

        This method has out-params. Python/C# wrappers look like:
        	`fz_image_resolution()` => `(int xres, int yres)`

        	Request the natural resolution
        	of an image.

        	xres, yres: Pointers to ints to be updated with the
        	natural resolution of an image (or a sensible default
        	if not encoded).
        """
        return _mupdf.FzImage_fz_image_resolution(self, xres, yres)

    def fz_image_size(self):
        """
        Class-aware wrapper for `::fz_image_size()`.
        	Return the size of the storage used by an image.
        """
        return _mupdf.FzImage_fz_image_size(self)

    def fz_new_buffer_from_image_as_jpeg(self, color_params, quality, invert_cmyk):
        """ Class-aware wrapper for `::fz_new_buffer_from_image_as_jpeg()`."""
        return _mupdf.FzImage_fz_new_buffer_from_image_as_jpeg(self, color_params, quality, invert_cmyk)

    def fz_new_buffer_from_image_as_jpx(self, color_params, quality):
        """ Class-aware wrapper for `::fz_new_buffer_from_image_as_jpx()`."""
        return _mupdf.FzImage_fz_new_buffer_from_image_as_jpx(self, color_params, quality)

    def fz_new_buffer_from_image_as_pam(self, color_params):
        """ Class-aware wrapper for `::fz_new_buffer_from_image_as_pam()`."""
        return _mupdf.FzImage_fz_new_buffer_from_image_as_pam(self, color_params)

    def fz_new_buffer_from_image_as_png(self, color_params):
        """
        Class-aware wrapper for `::fz_new_buffer_from_image_as_png()`.
        	Reencode a given image as a PNG into a buffer.

        	Ownership of the buffer is returned.
        """
        return _mupdf.FzImage_fz_new_buffer_from_image_as_png(self, color_params)

    def fz_new_buffer_from_image_as_pnm(self, color_params):
        """ Class-aware wrapper for `::fz_new_buffer_from_image_as_pnm()`."""
        return _mupdf.FzImage_fz_new_buffer_from_image_as_pnm(self, color_params)

    def fz_new_buffer_from_image_as_psd(self, color_params):
        """ Class-aware wrapper for `::fz_new_buffer_from_image_as_psd()`."""
        return _mupdf.FzImage_fz_new_buffer_from_image_as_psd(self, color_params)

    def __init__(self, *args):
        """
        *Overload 1:*
         == Constructors.  Constructor using `fz_new_image_from_buffer()`.
        		Create a new image from a
        		buffer of data, inferring its type from the format
        		of the data.


        |

        *Overload 2:*
         Constructor using `fz_new_image_from_compressed_buffer()`.
        		Create an image based on
        		the data in the supplied compressed buffer.

        		w,h: Width and height of the created image.

        		bpc: Bits per component.

        		colorspace: The colorspace (determines the number of components,
        		and any color conversions required while decoding).

        		xres, yres: The X and Y resolutions respectively.

        		interpolate: 1 if interpolation should be used when decoding
        		this image, 0 otherwise.

        		imagemask: 1 if this is an imagemask (i.e. transparency bitmap
        		mask), 0 otherwise.

        		decode: NULL, or a pointer to to a decode array. The default
        		decode array is [0 1] (repeated n times, for n color components).

        		colorkey: NULL, or a pointer to a colorkey array. The default
        		colorkey array is [0 255] (repeated n times, for n color
        		components).

        		buffer: Buffer of compressed data and compression parameters.
        		Ownership of this reference is passed in.

        		mask: NULL, or another image to use as a mask for this one.
        		A new reference is taken to this image. Supplying a masked
        		image as a mask to another image is illegal!


        |

        *Overload 3:*
         Constructor using `fz_new_image_from_compressed_buffer2()`.  Swig-friendly wrapper for fz_new_image_from_compressed_buffer(),
        	uses specified `decode` and `colorkey` if they are not null (in which
        	case we assert that they have size `2*fz_colorspace_n(colorspace)`).

        |

        *Overload 4:*
         Constructor using `fz_new_image_from_display_list()`.
        		Create a new image from a display list.

        		w, h: The conceptual width/height of the image.

        		transform: The matrix that needs to be applied to the given
        		list to make it render to the unit square.

        		list: The display list.


        |

        *Overload 5:*
         Constructor using `fz_new_image_from_file()`.
        		Create a new image from the contents
        		of a file, inferring its type from the format of the
        		data.


        |

        *Overload 6:*
         Constructor using `fz_new_image_from_pixmap()`.
        		Create an image from the given
        		pixmap.

        		pixmap: The pixmap to base the image upon. A new reference
        		to this is taken.

        		mask: NULL, or another image to use as a mask for this one.
        		A new reference is taken to this image. Supplying a masked
        		image as a mask to another image is illegal!


        |

        *Overload 7:*
         Constructor using `fz_new_image_from_svg()`.
        		Create a scalable image from an SVG document.


        |

        *Overload 8:*
         Constructor using `fz_new_image_from_svg_xml()`.
        		Create a scalable image from an SVG document.


        |

        *Overload 9:*
         Constructor using `fz_new_image_of_size()`.
        		Internal function to make a new fz_image structure
        		for a derived class.

        		w,h: Width and height of the created image.

        		bpc: Bits per component.

        		colorspace: The colorspace (determines the number of components,
        		and any color conversions required while decoding).

        		xres, yres: The X and Y resolutions respectively.

        		interpolate: 1 if interpolation should be used when decoding
        		this image, 0 otherwise.

        		imagemask: 1 if this is an imagemask (i.e. transparent), 0
        		otherwise.

        		decode: NULL, or a pointer to to a decode array. The default
        		decode array is [0 1] (repeated n times, for n color components).

        		colorkey: NULL, or a pointer to a colorkey array. The default
        		colorkey array is [0 255] (repeated n times, for n color
        		components).

        		mask: NULL, or another image to use as a mask for this one.
        		A new reference is taken to this image. Supplying a masked
        		image as a mask to another image is illegal!

        		size: The size of the required allocated structure (the size of
        		the derived structure).

        		get: The function to be called to obtain a decoded pixmap.

        		get_size: The function to be called to return the storage size
        		used by this image.

        		drop: The function to be called to dispose of this image once
        		the last reference is dropped.

        		Returns a pointer to an allocated structure of the required size,
        		with the first sizeof(fz_image) bytes initialised as appropriate
        		given the supplied parameters, and the other bytes set to zero.


        |

        *Overload 10:*
         Copy constructor using `fz_keep_image()`.

        |

        *Overload 11:*
         Default constructor, sets `m_internal` to null.

        |

        *Overload 12:*
         Constructor using raw copy of pre-existing `::fz_image`.
        """
        _mupdf.FzImage_swiginit(self, _mupdf.new_FzImage(*args))

    def key_storable(self):
        return _mupdf.FzImage_key_storable(self)

    def w(self):
        return _mupdf.FzImage_w(self)

    def h(self):
        return _mupdf.FzImage_h(self)

    def n(self):
        return _mupdf.FzImage_n(self)

    def bpc(self):
        return _mupdf.FzImage_bpc(self)

    def imagemask(self):
        return _mupdf.FzImage_imagemask(self)

    def interpolate(self):
        return _mupdf.FzImage_interpolate(self)

    def use_colorkey(self):
        return _mupdf.FzImage_use_colorkey(self)

    def use_decode(self):
        return _mupdf.FzImage_use_decode(self)

    def decoded(self):
        return _mupdf.FzImage_decoded(self)

    def scalable(self):
        return _mupdf.FzImage_scalable(self)

    def orientation(self):
        return _mupdf.FzImage_orientation(self)

    def mask(self):
        return _mupdf.FzImage_mask(self)

    def xres(self):
        return _mupdf.FzImage_xres(self)

    def yres(self):
        return _mupdf.FzImage_yres(self)

    def colorspace(self):
        return _mupdf.FzImage_colorspace(self)

    def colorkey(self):
        return _mupdf.FzImage_colorkey(self)

    def decode(self):
        return _mupdf.FzImage_decode(self)
    __swig_destroy__ = _mupdf.delete_FzImage

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzImage_m_internal_value(self)
    m_internal = property(_mupdf.FzImage_m_internal_get, _mupdf.FzImage_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzImage_s_num_instances_get, _mupdf.FzImage_s_num_instances_set)