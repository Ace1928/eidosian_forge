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
class FzPixmap(object):
    """
    Wrapper class for struct `fz_pixmap`.
    Pixmaps represent a set of pixels for a 2 dimensional region of
    a plane. Each pixel has n components per pixel. The components
    are in the order process-components, spot-colors, alpha, where
    there can be 0 of any of those types. The data is in
    premultiplied alpha when rendering, but non-premultiplied for
    colorspace conversions and rescaling.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    @staticmethod
    def pdf_new_pixmap_from_page_with_usage(page, ctm, cs, alpha, usage, box):
        """ Class-aware wrapper for `::pdf_new_pixmap_from_page_with_usage()`."""
        return _mupdf.FzPixmap_pdf_new_pixmap_from_page_with_usage(page, ctm, cs, alpha, usage, box)

    @staticmethod
    def pdf_new_pixmap_from_page_with_separations_and_usage(page, ctm, cs, seps, alpha, usage, box):
        """ Class-aware wrapper for `::pdf_new_pixmap_from_page_with_separations_and_usage()`."""
        return _mupdf.FzPixmap_pdf_new_pixmap_from_page_with_separations_and_usage(page, ctm, cs, seps, alpha, usage, box)

    @staticmethod
    def fz_new_pixmap_from_page_contents(page, ctm, cs, alpha):
        """
        Class-aware wrapper for `::fz_new_pixmap_from_page_contents()`.
        	Render the page contents without annotations.

        	Ownership of the pixmap is returned to the caller.
        """
        return _mupdf.FzPixmap_fz_new_pixmap_from_page_contents(page, ctm, cs, alpha)

    @staticmethod
    def fz_new_pixmap_from_page_contents_with_separations(page, ctm, cs, seps, alpha):
        """ Class-aware wrapper for `::fz_new_pixmap_from_page_contents_with_separations()`."""
        return _mupdf.FzPixmap_fz_new_pixmap_from_page_contents_with_separations(page, ctm, cs, seps, alpha)

    def fz_alpha_from_gray(self):
        """ Class-aware wrapper for `::fz_alpha_from_gray()`."""
        return _mupdf.FzPixmap_fz_alpha_from_gray(self)

    def fz_clear_pixmap(self):
        """
        Class-aware wrapper for `::fz_clear_pixmap()`.
        	Sets all components (including alpha) of
        	all pixels in a pixmap to 0.

        	pix: The pixmap to clear.
        """
        return _mupdf.FzPixmap_fz_clear_pixmap(self)

    def fz_clear_pixmap_rect_with_value(self, value, r):
        """
        Class-aware wrapper for `::fz_clear_pixmap_rect_with_value()`.
        	Clears a subrect of a pixmap with the given value.

        	pix: The pixmap to clear.

        	value: Values in the range 0 to 255 are valid. Each component
        	sample for each pixel in the pixmap will be set to this value,
        	while alpha will always be set to 255 (non-transparent).

        	r: the rectangle.
        """
        return _mupdf.FzPixmap_fz_clear_pixmap_rect_with_value(self, value, r)

    def fz_clear_pixmap_with_value(self, value):
        """
        Class-aware wrapper for `::fz_clear_pixmap_with_value()`.
        	Clears a pixmap with the given value.

        	pix: The pixmap to clear.

        	value: Values in the range 0 to 255 are valid. Each component
        	sample for each pixel in the pixmap will be set to this value,
        	while alpha will always be set to 255 (non-transparent).

        	This function is horrible, and should be removed from the
        	API and replaced with a less magic one.
        """
        return _mupdf.FzPixmap_fz_clear_pixmap_with_value(self, value)

    def fz_clone_pixmap(self):
        """
        Class-aware wrapper for `::fz_clone_pixmap()`.
        	Clone a pixmap, copying the pixels and associated data to new
        	storage.

        	The reference count of 'old' is unchanged.
        """
        return _mupdf.FzPixmap_fz_clone_pixmap(self)

    def fz_clone_pixmap_area_with_different_seps(self, bbox, dcs, seps, color_params, default_cs):
        """ Class-aware wrapper for `::fz_clone_pixmap_area_with_different_seps()`."""
        return _mupdf.FzPixmap_fz_clone_pixmap_area_with_different_seps(self, bbox, dcs, seps, color_params, default_cs)

    def fz_convert_indexed_pixmap_to_base(self):
        """
        Class-aware wrapper for `::fz_convert_indexed_pixmap_to_base()`.
        	Convert pixmap from indexed to base colorspace.

        	This creates a new bitmap containing the converted pixmap data.
        """
        return _mupdf.FzPixmap_fz_convert_indexed_pixmap_to_base(self)

    def fz_convert_pixmap(self, cs_des, prf, default_cs, color_params, keep_alpha):
        """
        Class-aware wrapper for `::fz_convert_pixmap()`.
        	Convert an existing pixmap to a desired
        	colorspace. Other properties of the pixmap, such as resolution
        	and position are copied to the converted pixmap.

        	pix: The pixmap to convert.

        	default_cs: If NULL pix->colorspace is used. It is possible that
        	the data may need to be interpreted as one of the color spaces
        	in default_cs.

        	cs_des: Desired colorspace, may be NULL to denote alpha-only.

        	prf: Proofing color space through which we need to convert.

        	color_params: Parameters that may be used in conversion (e.g.
        	ri).

        	keep_alpha: If 0 any alpha component is removed, otherwise
        	alpha is kept if present in the pixmap.
        """
        return _mupdf.FzPixmap_fz_convert_pixmap(self, cs_des, prf, default_cs, color_params, keep_alpha)

    def fz_convert_separation_pixmap_to_base(self):
        """
        Class-aware wrapper for `::fz_convert_separation_pixmap_to_base()`.
        	Convert pixmap from DeviceN/Separation to base colorspace.

        	This creates a new bitmap containing the converted pixmap data.
        """
        return _mupdf.FzPixmap_fz_convert_separation_pixmap_to_base(self)

    def fz_copy_pixmap_rect(self, src, r, default_cs):
        """ Class-aware wrapper for `::fz_copy_pixmap_rect()`."""
        return _mupdf.FzPixmap_fz_copy_pixmap_rect(self, src, r, default_cs)

    def fz_decode_tile(self, decode):
        """ Class-aware wrapper for `::fz_decode_tile()`."""
        return _mupdf.FzPixmap_fz_decode_tile(self, decode)

    def fz_fill_pixmap_with_color(self, colorspace, color, color_params):
        """
        Class-aware wrapper for `::fz_fill_pixmap_with_color()`.

        This method has out-params. Python/C# wrappers look like:
        	`fz_fill_pixmap_with_color(::fz_colorspace *colorspace, ::fz_color_params color_params)` => float color

        	Fill pixmap with solid color.
        """
        return _mupdf.FzPixmap_fz_fill_pixmap_with_color(self, colorspace, color, color_params)

    def fz_gamma_pixmap(self, gamma):
        """
        Class-aware wrapper for `::fz_gamma_pixmap()`.
        	Apply gamma correction to a pixmap. All components
        	of all pixels are modified (except alpha, which is unchanged).

        	gamma: The gamma value to apply; 1.0 for no change.
        """
        return _mupdf.FzPixmap_fz_gamma_pixmap(self, gamma)

    def fz_generate_transition(self, opix, npix, time, trans):
        """
        Class-aware wrapper for `::fz_generate_transition()`.
        	Generate a frame of a transition.

        	tpix: Target pixmap
        	opix: Old pixmap
        	npix: New pixmap
        	time: Position within the transition (0 to 256)
        	trans: Transition details

        	Returns 1 if successfully generated a frame.

        	Note: Pixmaps must include alpha.
        """
        return _mupdf.FzPixmap_fz_generate_transition(self, opix, npix, time, trans)

    def fz_invert_pixmap(self):
        """
        Class-aware wrapper for `::fz_invert_pixmap()`.
        	Invert all the pixels in a pixmap. All components (process and
        	spots) of all pixels are inverted (except alpha, which is
        	unchanged).
        """
        return _mupdf.FzPixmap_fz_invert_pixmap(self)

    def fz_invert_pixmap_alpha(self):
        """
        Class-aware wrapper for `::fz_invert_pixmap_alpha()`.
        	Invert the alpha fo all the pixels in a pixmap.
        """
        return _mupdf.FzPixmap_fz_invert_pixmap_alpha(self)

    def fz_invert_pixmap_luminance(self):
        """
        Class-aware wrapper for `::fz_invert_pixmap_luminance()`.
        	Transform the pixels in a pixmap so that luminance of each
        	pixel is inverted, and the chrominance remains unchanged (as
        	much as accuracy allows).

        	All components of all pixels are inverted (except alpha, which
        	is unchanged). Only supports Grey and RGB bitmaps.
        """
        return _mupdf.FzPixmap_fz_invert_pixmap_luminance(self)

    def fz_invert_pixmap_raw(self):
        """
        Class-aware wrapper for `::fz_invert_pixmap_raw()`.
        	Invert all the pixels in a non-premultiplied pixmap in a
        	very naive manner.
        """
        return _mupdf.FzPixmap_fz_invert_pixmap_raw(self)

    def fz_invert_pixmap_rect(self, rect):
        """
        Class-aware wrapper for `::fz_invert_pixmap_rect()`.
        	Invert all the pixels in a given rectangle of a (premultiplied)
        	pixmap. All components of all pixels in the rectangle are
        	inverted (except alpha, which is unchanged).
        """
        return _mupdf.FzPixmap_fz_invert_pixmap_rect(self, rect)

    def fz_is_pixmap_monochrome(self):
        """
        Class-aware wrapper for `::fz_is_pixmap_monochrome()`.
        	Check if the pixmap is a 1-channel image containing samples with
        	only values 0 and 255
        """
        return _mupdf.FzPixmap_fz_is_pixmap_monochrome(self)

    def fz_md5_pixmap(self, digest):
        """ Class-aware wrapper for `::fz_md5_pixmap()`."""
        return _mupdf.FzPixmap_fz_md5_pixmap(self, digest)

    def fz_md5_pixmap2(self):
        """
        Class-aware wrapper for `::fz_md5_pixmap2()`.
        C++ alternative to `fz_md5_pixmap()` that returns the digest by value.
        """
        return _mupdf.FzPixmap_fz_md5_pixmap2(self)

    def fz_new_bitmap_from_pixmap(self, ht):
        """
        Class-aware wrapper for `::fz_new_bitmap_from_pixmap()`.
        	Make a bitmap from a pixmap and a halftone.

        	pix: The pixmap to generate from. Currently must be a single
        	color component with no alpha.

        	ht: The halftone to use. NULL implies the default halftone.

        	Returns the resultant bitmap. Throws exceptions in the case of
        	failure to allocate.
        """
        return _mupdf.FzPixmap_fz_new_bitmap_from_pixmap(self, ht)

    def fz_new_bitmap_from_pixmap_band(self, ht, band_start):
        """
        Class-aware wrapper for `::fz_new_bitmap_from_pixmap_band()`.
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
        """
        return _mupdf.FzPixmap_fz_new_bitmap_from_pixmap_band(self, ht, band_start)

    def fz_new_buffer_from_pixmap_as_jpeg(self, color_params, quality, invert_cmyk):
        """ Class-aware wrapper for `::fz_new_buffer_from_pixmap_as_jpeg()`."""
        return _mupdf.FzPixmap_fz_new_buffer_from_pixmap_as_jpeg(self, color_params, quality, invert_cmyk)

    def fz_new_buffer_from_pixmap_as_jpx(self, color_params, quality):
        """ Class-aware wrapper for `::fz_new_buffer_from_pixmap_as_jpx()`."""
        return _mupdf.FzPixmap_fz_new_buffer_from_pixmap_as_jpx(self, color_params, quality)

    def fz_new_buffer_from_pixmap_as_pam(self, color_params):
        """ Class-aware wrapper for `::fz_new_buffer_from_pixmap_as_pam()`."""
        return _mupdf.FzPixmap_fz_new_buffer_from_pixmap_as_pam(self, color_params)

    def fz_new_buffer_from_pixmap_as_png(self, color_params):
        """
        Class-aware wrapper for `::fz_new_buffer_from_pixmap_as_png()`.
        	Reencode a given pixmap as a PNG into a buffer.

        	Ownership of the buffer is returned.
        """
        return _mupdf.FzPixmap_fz_new_buffer_from_pixmap_as_png(self, color_params)

    def fz_new_buffer_from_pixmap_as_pnm(self, color_params):
        """ Class-aware wrapper for `::fz_new_buffer_from_pixmap_as_pnm()`."""
        return _mupdf.FzPixmap_fz_new_buffer_from_pixmap_as_pnm(self, color_params)

    def fz_new_buffer_from_pixmap_as_psd(self, color_params):
        """ Class-aware wrapper for `::fz_new_buffer_from_pixmap_as_psd()`."""
        return _mupdf.FzPixmap_fz_new_buffer_from_pixmap_as_psd(self, color_params)

    def fz_new_image_from_pixmap(self, mask):
        """
        Class-aware wrapper for `::fz_new_image_from_pixmap()`.
        	Create an image from the given
        	pixmap.

        	pixmap: The pixmap to base the image upon. A new reference
        	to this is taken.

        	mask: NULL, or another image to use as a mask for this one.
        	A new reference is taken to this image. Supplying a masked
        	image as a mask to another image is illegal!
        """
        return _mupdf.FzPixmap_fz_new_image_from_pixmap(self, mask)

    def fz_new_pixmap_from_alpha_channel(self):
        """ Class-aware wrapper for `::fz_new_pixmap_from_alpha_channel()`."""
        return _mupdf.FzPixmap_fz_new_pixmap_from_alpha_channel(self)

    def fz_new_pixmap_from_color_and_mask(self, mask):
        """ Class-aware wrapper for `::fz_new_pixmap_from_color_and_mask()`."""
        return _mupdf.FzPixmap_fz_new_pixmap_from_color_and_mask(self, mask)

    def fz_new_pixmap_from_pixmap(self, rect):
        """
        Class-aware wrapper for `::fz_new_pixmap_from_pixmap()`.
        	Create a new pixmap that represents a subarea of the specified
        	pixmap. A reference is taken to this pixmap that will be dropped
        	on destruction.

        	The supplied rectangle must be wholly contained within the
        	original pixmap.

        	Returns a pointer to the new pixmap. Throws exception on failure
        	to allocate.
        """
        return _mupdf.FzPixmap_fz_new_pixmap_from_pixmap(self, rect)

    def fz_pixmap_alpha(self):
        """
        Class-aware wrapper for `::fz_pixmap_alpha()`.
        	Return the number of alpha planes in a pixmap.

        	Returns the number of alphas. Does not throw exceptions.
        """
        return _mupdf.FzPixmap_fz_pixmap_alpha(self)

    def fz_pixmap_bbox(self):
        """
        Class-aware wrapper for `::fz_pixmap_bbox()`.
        	Return the bounding box for a pixmap.
        """
        return _mupdf.FzPixmap_fz_pixmap_bbox(self)

    def fz_pixmap_colorants(self):
        """
        Class-aware wrapper for `::fz_pixmap_colorants()`.
        	Return the number of colorants in a pixmap.

        	Returns the number of colorants (components, less any spots and
        	alpha).
        """
        return _mupdf.FzPixmap_fz_pixmap_colorants(self)

    def fz_pixmap_colorspace(self):
        """
        Class-aware wrapper for `::fz_pixmap_colorspace()`.
        	Return the colorspace of a pixmap

        	Returns colorspace.
        """
        return _mupdf.FzPixmap_fz_pixmap_colorspace(self)

    def fz_pixmap_components(self):
        """
        Class-aware wrapper for `::fz_pixmap_components()`.
        	Return the number of components in a pixmap.

        	Returns the number of components (including spots and alpha).
        """
        return _mupdf.FzPixmap_fz_pixmap_components(self)

    def fz_pixmap_height(self):
        """
        Class-aware wrapper for `::fz_pixmap_height()`.
        	Return the height of the pixmap in pixels.
        """
        return _mupdf.FzPixmap_fz_pixmap_height(self)

    def fz_pixmap_samples(self):
        """
        Class-aware wrapper for `::fz_pixmap_samples()`.
        	Returns a pointer to the pixel data of a pixmap.

        	Returns the pointer.
        """
        return _mupdf.FzPixmap_fz_pixmap_samples(self)

    def fz_pixmap_samples_int(self):
        """ Class-aware wrapper for `::fz_pixmap_samples_int()`."""
        return _mupdf.FzPixmap_fz_pixmap_samples_int(self)

    def fz_pixmap_size(self):
        """
        Class-aware wrapper for `::fz_pixmap_size()`.
        	Return sizeof fz_pixmap plus size of data, in bytes.
        """
        return _mupdf.FzPixmap_fz_pixmap_size(self)

    def fz_pixmap_spots(self):
        """
        Class-aware wrapper for `::fz_pixmap_spots()`.
        	Return the number of spots in a pixmap.

        	Returns the number of spots (components, less colorants and
        	alpha). Does not throw exceptions.
        """
        return _mupdf.FzPixmap_fz_pixmap_spots(self)

    def fz_pixmap_stride(self):
        """
        Class-aware wrapper for `::fz_pixmap_stride()`.
        	Return the number of bytes in a row in the pixmap.
        """
        return _mupdf.FzPixmap_fz_pixmap_stride(self)

    def fz_pixmap_width(self):
        """
        Class-aware wrapper for `::fz_pixmap_width()`.
        	Return the width of the pixmap in pixels.
        """
        return _mupdf.FzPixmap_fz_pixmap_width(self)

    def fz_pixmap_x(self):
        """
        Class-aware wrapper for `::fz_pixmap_x()`.
        	Return the x value of the pixmap in pixels.
        """
        return _mupdf.FzPixmap_fz_pixmap_x(self)

    def fz_pixmap_y(self):
        """
        Class-aware wrapper for `::fz_pixmap_y()`.
        	Return the y value of the pixmap in pixels.
        """
        return _mupdf.FzPixmap_fz_pixmap_y(self)

    def fz_samples_get(self, offset):
        """
        Class-aware wrapper for `::fz_samples_get()`.
        Provides simple (but slow) access to pixmap data from Python and C#.
        """
        return _mupdf.FzPixmap_fz_samples_get(self, offset)

    def fz_samples_set(self, offset, value):
        """
        Class-aware wrapper for `::fz_samples_set()`.
        Provides simple (but slow) write access to pixmap data from Python and
        C#.
        """
        return _mupdf.FzPixmap_fz_samples_set(self, offset, value)

    def fz_save_pixmap_as_jpeg(self, filename, quality):
        """
        Class-aware wrapper for `::fz_save_pixmap_as_jpeg()`.
        	Save a pixmap as a JPEG.
        """
        return _mupdf.FzPixmap_fz_save_pixmap_as_jpeg(self, filename, quality)

    def fz_save_pixmap_as_jpx(self, filename, q):
        """
        Class-aware wrapper for `::fz_save_pixmap_as_jpx()`.
        	Save pixmap data as JP2K with no subsampling.

        	quality = 100 = lossless
        	otherwise for a factor of x compression use 100-x. (so 80 is 1:20 compression)
        """
        return _mupdf.FzPixmap_fz_save_pixmap_as_jpx(self, filename, q)

    def fz_save_pixmap_as_pam(self, filename):
        """
        Class-aware wrapper for `::fz_save_pixmap_as_pam()`.
        	Save a pixmap as a pnm (greyscale, rgb or cmyk, with or without
        	alpha).
        """
        return _mupdf.FzPixmap_fz_save_pixmap_as_pam(self, filename)

    def fz_save_pixmap_as_pbm(self, filename):
        """
        Class-aware wrapper for `::fz_save_pixmap_as_pbm()`.
        	Save a pixmap as a pbm. (Performing halftoning).
        """
        return _mupdf.FzPixmap_fz_save_pixmap_as_pbm(self, filename)

    def fz_save_pixmap_as_pcl(self, filename, append, pcl):
        """
        Class-aware wrapper for `::fz_save_pixmap_as_pcl()`.
        	Save an (RGB) pixmap as color PCL.
        """
        return _mupdf.FzPixmap_fz_save_pixmap_as_pcl(self, filename, append, pcl)

    def fz_save_pixmap_as_pclm(self, filename, append, options):
        """
        Class-aware wrapper for `::fz_save_pixmap_as_pclm()`.
        	Save a (Greyscale or RGB) pixmap as pclm.
        """
        return _mupdf.FzPixmap_fz_save_pixmap_as_pclm(self, filename, append, options)

    def fz_save_pixmap_as_pdfocr(self, filename, append, options):
        """
        Class-aware wrapper for `::fz_save_pixmap_as_pdfocr()`.
        	Save a (Greyscale or RGB) pixmap as pdfocr.
        """
        return _mupdf.FzPixmap_fz_save_pixmap_as_pdfocr(self, filename, append, options)

    def fz_save_pixmap_as_pkm(self, filename):
        """
        Class-aware wrapper for `::fz_save_pixmap_as_pkm()`.
        	Save a CMYK pixmap as a pkm. (Performing halftoning).
        """
        return _mupdf.FzPixmap_fz_save_pixmap_as_pkm(self, filename)

    def fz_save_pixmap_as_png(self, filename):
        """
        Class-aware wrapper for `::fz_save_pixmap_as_png()`.
        	Save a (Greyscale or RGB) pixmap as a png.
        """
        return _mupdf.FzPixmap_fz_save_pixmap_as_png(self, filename)

    def fz_save_pixmap_as_pnm(self, filename):
        """
        Class-aware wrapper for `::fz_save_pixmap_as_pnm()`.
        	Save a pixmap as a pnm (greyscale or rgb, no alpha).
        """
        return _mupdf.FzPixmap_fz_save_pixmap_as_pnm(self, filename)

    def fz_save_pixmap_as_ps(self, filename, append):
        """
        Class-aware wrapper for `::fz_save_pixmap_as_ps()`.
        	Save a (gray, rgb, or cmyk, no alpha) pixmap out as postscript.
        """
        return _mupdf.FzPixmap_fz_save_pixmap_as_ps(self, filename, append)

    def fz_save_pixmap_as_psd(self, filename):
        """
        Class-aware wrapper for `::fz_save_pixmap_as_psd()`.
        	Save a pixmap as a PSD file.
        """
        return _mupdf.FzPixmap_fz_save_pixmap_as_psd(self, filename)

    def fz_save_pixmap_as_pwg(self, filename, append, pwg):
        """
        Class-aware wrapper for `::fz_save_pixmap_as_pwg()`.
        	Save a pixmap as a PWG.
        """
        return _mupdf.FzPixmap_fz_save_pixmap_as_pwg(self, filename, append, pwg)

    def fz_scale_pixmap(self, x, y, w, h, clip):
        """ Class-aware wrapper for `::fz_scale_pixmap()`."""
        return _mupdf.FzPixmap_fz_scale_pixmap(self, x, y, w, h, clip)

    def fz_set_pixmap_resolution(self, xres, yres):
        """
        Class-aware wrapper for `::fz_set_pixmap_resolution()`.
        	Set the pixels per inch resolution of the pixmap.
        """
        return _mupdf.FzPixmap_fz_set_pixmap_resolution(self, xres, yres)

    def fz_subsample_pixmap(self, factor):
        """ Class-aware wrapper for `::fz_subsample_pixmap()`."""
        return _mupdf.FzPixmap_fz_subsample_pixmap(self, factor)

    def fz_tint_pixmap(self, black, white):
        """
        Class-aware wrapper for `::fz_tint_pixmap()`.
        	Tint all the pixels in an RGB, BGR, or Gray pixmap.

        	black: Map black to this hexadecimal RGB color.

        	white: Map white to this hexadecimal RGB color.
        """
        return _mupdf.FzPixmap_fz_tint_pixmap(self, black, white)

    def __init__(self, *args):
        """
        *Overload 1:*
         == Constructors.  Constructor using `fz_new_pixmap()`.
        		Create a new pixmap, with its origin at (0,0)

        		cs: The colorspace to use for the pixmap, or NULL for an alpha
        		plane/mask.

        		w: The width of the pixmap (in pixels)

        		h: The height of the pixmap (in pixels)

        		seps: Details of separations.

        		alpha: 0 for no alpha, 1 for alpha.

        		Returns a pointer to the new pixmap. Throws exception on failure
        		to allocate.


        |

        *Overload 2:*
         Constructor using `fz_new_pixmap_from_color_and_mask()`.

        |

        *Overload 3:*
         Constructor using `fz_new_pixmap_from_display_list()`.
        		Render the page to a pixmap using the transform and colorspace.

        		Ownership of the pixmap is returned to the caller.


        |

        *Overload 4:*
         Constructor using `fz_new_pixmap_from_display_list_with_separations()`.
        		Render the page contents with control over spot colors.

        		Ownership of the pixmap is returned to the caller.


        |

        *Overload 5:*
         Constructor using `fz_new_pixmap_from_page()`.

        |

        *Overload 6:*
         Constructor using `fz_new_pixmap_from_page_number()`.

        |

        *Overload 7:*
         Constructor using `fz_new_pixmap_from_page_number_with_separations()`.

        |

        *Overload 8:*
         Constructor using `fz_new_pixmap_from_page_with_separations()`.

        |

        *Overload 9:*
         Constructor using `fz_new_pixmap_from_pixmap()`.
        		Create a new pixmap that represents a subarea of the specified
        		pixmap. A reference is taken to this pixmap that will be dropped
        		on destruction.

        		The supplied rectangle must be wholly contained within the
        		original pixmap.

        		Returns a pointer to the new pixmap. Throws exception on failure
        		to allocate.


        |

        *Overload 10:*
         Constructor using `fz_new_pixmap_with_bbox()`.
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


        |

        *Overload 11:*
         Constructor using `fz_new_pixmap_with_bbox_and_data()`.
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


        |

        *Overload 12:*
         Constructor using `fz_new_pixmap_with_data()`.
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


        |

        *Overload 13:*
         Constructor using `pdf_new_pixmap_from_annot()`.

        |

        *Overload 14:*
         Constructor using `pdf_new_pixmap_from_page_contents_with_separations_and_usage()`.

        |

        *Overload 15:*
         Constructor using `pdf_new_pixmap_from_page_contents_with_usage()`.

        |

        *Overload 16:*
         Copy constructor using `fz_keep_pixmap()`.

        |

        *Overload 17:*
         Default constructor, sets `m_internal` to null.

        |

        *Overload 18:*
         Constructor using raw copy of pre-existing `::fz_pixmap`.
        """
        _mupdf.FzPixmap_swiginit(self, _mupdf.new_FzPixmap(*args))

    def storable(self):
        return _mupdf.FzPixmap_storable(self)

    def x(self):
        return _mupdf.FzPixmap_x(self)

    def y(self):
        return _mupdf.FzPixmap_y(self)

    def w(self):
        return _mupdf.FzPixmap_w(self)

    def h(self):
        return _mupdf.FzPixmap_h(self)

    def n(self):
        return _mupdf.FzPixmap_n(self)

    def s(self):
        return _mupdf.FzPixmap_s(self)

    def alpha(self):
        return _mupdf.FzPixmap_alpha(self)

    def flags(self):
        return _mupdf.FzPixmap_flags(self)

    def stride(self):
        return _mupdf.FzPixmap_stride(self)

    def seps(self):
        return _mupdf.FzPixmap_seps(self)

    def xres(self):
        return _mupdf.FzPixmap_xres(self)

    def yres(self):
        return _mupdf.FzPixmap_yres(self)

    def colorspace(self):
        return _mupdf.FzPixmap_colorspace(self)

    def samples(self):
        return _mupdf.FzPixmap_samples(self)

    def underlying(self):
        return _mupdf.FzPixmap_underlying(self)
    __swig_destroy__ = _mupdf.delete_FzPixmap

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzPixmap_m_internal_value(self)
    m_internal = property(_mupdf.FzPixmap_m_internal_get, _mupdf.FzPixmap_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzPixmap_s_num_instances_get, _mupdf.FzPixmap_s_num_instances_set)