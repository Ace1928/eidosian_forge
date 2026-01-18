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
class FzDevice(object):
    """
    Wrapper class for struct `fz_device`.
    The different format handlers (pdf, xps etc) interpret pages to
    a device. These devices can then process the stream of calls
    they receive in various ways:
    	The trace device outputs debugging information for the calls.
    	The draw device will render them.
    	The list device stores them in a list to play back later.
    	The text device performs text extraction and searching.
    	The bbox device calculates the bounding box for the page.
    Other devices can (and will) be written in the future.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    @staticmethod
    def fz_new_xmltext_device(out):
        """
        Class-aware wrapper for `::fz_new_xmltext_device()`.
        	Create a device to output raw information.
        """
        return _mupdf.FzDevice_fz_new_xmltext_device(out)

    @staticmethod
    def fz_new_draw_device_type3(transform, dest):
        """ Class-aware wrapper for `::fz_new_draw_device_type3()`."""
        return _mupdf.FzDevice_fz_new_draw_device_type3(transform, dest)

    def fz_begin_group(self, area, cs, isolated, knockout, blendmode, alpha):
        """ Class-aware wrapper for `::fz_begin_group()`."""
        return _mupdf.FzDevice_fz_begin_group(self, area, cs, isolated, knockout, blendmode, alpha)

    def fz_begin_layer(self, layer_name):
        """ Class-aware wrapper for `::fz_begin_layer()`."""
        return _mupdf.FzDevice_fz_begin_layer(self, layer_name)

    def fz_begin_mask(self, area, luminosity, colorspace, bc, color_params):
        """ Class-aware wrapper for `::fz_begin_mask()`."""
        return _mupdf.FzDevice_fz_begin_mask(self, area, luminosity, colorspace, bc, color_params)

    def fz_begin_metatext(self, meta, text):
        """ Class-aware wrapper for `::fz_begin_metatext()`."""
        return _mupdf.FzDevice_fz_begin_metatext(self, meta, text)

    def fz_begin_structure(self, standard, raw, idx):
        """ Class-aware wrapper for `::fz_begin_structure()`."""
        return _mupdf.FzDevice_fz_begin_structure(self, standard, raw, idx)

    def fz_begin_tile(self, area, view, xstep, ystep, ctm):
        """ Class-aware wrapper for `::fz_begin_tile()`."""
        return _mupdf.FzDevice_fz_begin_tile(self, area, view, xstep, ystep, ctm)

    def fz_begin_tile_id(self, area, view, xstep, ystep, ctm, id):
        """ Class-aware wrapper for `::fz_begin_tile_id()`."""
        return _mupdf.FzDevice_fz_begin_tile_id(self, area, view, xstep, ystep, ctm, id)

    def fz_clip_image_mask(self, image, ctm, scissor):
        """ Class-aware wrapper for `::fz_clip_image_mask()`."""
        return _mupdf.FzDevice_fz_clip_image_mask(self, image, ctm, scissor)

    def fz_clip_path(self, path, even_odd, ctm, scissor):
        """ Class-aware wrapper for `::fz_clip_path()`."""
        return _mupdf.FzDevice_fz_clip_path(self, path, even_odd, ctm, scissor)

    def fz_clip_stroke_path(self, path, stroke, ctm, scissor):
        """ Class-aware wrapper for `::fz_clip_stroke_path()`."""
        return _mupdf.FzDevice_fz_clip_stroke_path(self, path, stroke, ctm, scissor)

    def fz_clip_stroke_text(self, text, stroke, ctm, scissor):
        """ Class-aware wrapper for `::fz_clip_stroke_text()`."""
        return _mupdf.FzDevice_fz_clip_stroke_text(self, text, stroke, ctm, scissor)

    def fz_clip_text(self, text, ctm, scissor):
        """ Class-aware wrapper for `::fz_clip_text()`."""
        return _mupdf.FzDevice_fz_clip_text(self, text, ctm, scissor)

    def fz_close_device(self):
        """
        Class-aware wrapper for `::fz_close_device()`.
        	Signal the end of input, and flush any buffered output.
        	This is NOT called implicitly on fz_drop_device. This
        	may throw exceptions.
        """
        return _mupdf.FzDevice_fz_close_device(self)

    def fz_device_current_scissor(self):
        """
        Class-aware wrapper for `::fz_device_current_scissor()`.
        	Find current scissor region as tracked by the device.
        """
        return _mupdf.FzDevice_fz_device_current_scissor(self)

    def fz_disable_device_hints(self, hints):
        """
        Class-aware wrapper for `::fz_disable_device_hints()`.
        	Disable (clear) hint bits within the hint bitfield for a device.
        """
        return _mupdf.FzDevice_fz_disable_device_hints(self, hints)

    def fz_enable_device_hints(self, hints):
        """
        Class-aware wrapper for `::fz_enable_device_hints()`.
        	Enable (set) hint bits within the hint bitfield for a device.
        """
        return _mupdf.FzDevice_fz_enable_device_hints(self, hints)

    def fz_end_group(self):
        """ Class-aware wrapper for `::fz_end_group()`."""
        return _mupdf.FzDevice_fz_end_group(self)

    def fz_end_layer(self):
        """ Class-aware wrapper for `::fz_end_layer()`."""
        return _mupdf.FzDevice_fz_end_layer(self)

    def fz_end_mask(self):
        """ Class-aware wrapper for `::fz_end_mask()`."""
        return _mupdf.FzDevice_fz_end_mask(self)

    def fz_end_mask_tr(self, fn):
        """ Class-aware wrapper for `::fz_end_mask_tr()`."""
        return _mupdf.FzDevice_fz_end_mask_tr(self, fn)

    def fz_end_metatext(self):
        """ Class-aware wrapper for `::fz_end_metatext()`."""
        return _mupdf.FzDevice_fz_end_metatext(self)

    def fz_end_structure(self):
        """ Class-aware wrapper for `::fz_end_structure()`."""
        return _mupdf.FzDevice_fz_end_structure(self)

    def fz_end_tile(self):
        """ Class-aware wrapper for `::fz_end_tile()`."""
        return _mupdf.FzDevice_fz_end_tile(self)

    def fz_fill_image(self, image, ctm, alpha, color_params):
        """ Class-aware wrapper for `::fz_fill_image()`."""
        return _mupdf.FzDevice_fz_fill_image(self, image, ctm, alpha, color_params)

    def fz_fill_image_mask(self, image, ctm, colorspace, color, alpha, color_params):
        """ Class-aware wrapper for `::fz_fill_image_mask()`."""
        return _mupdf.FzDevice_fz_fill_image_mask(self, image, ctm, colorspace, color, alpha, color_params)

    def fz_fill_path(self, path, even_odd, ctm, colorspace, color, alpha, color_params):
        """
        Class-aware wrapper for `::fz_fill_path()`.
        	Device calls; graphics primitives and containers.
        """
        return _mupdf.FzDevice_fz_fill_path(self, path, even_odd, ctm, colorspace, color, alpha, color_params)

    def fz_fill_shade(self, shade, ctm, alpha, color_params):
        """ Class-aware wrapper for `::fz_fill_shade()`."""
        return _mupdf.FzDevice_fz_fill_shade(self, shade, ctm, alpha, color_params)

    def fz_fill_text(self, text, ctm, colorspace, color, alpha, color_params):
        """ Class-aware wrapper for `::fz_fill_text()`."""
        return _mupdf.FzDevice_fz_fill_text(self, text, ctm, colorspace, color, alpha, color_params)

    def fz_ignore_text(self, text, ctm):
        """ Class-aware wrapper for `::fz_ignore_text()`."""
        return _mupdf.FzDevice_fz_ignore_text(self, text, ctm)

    def fz_new_ocr_device(self, ctm, mediabox, with_list, language, datadir, progress, progress_arg):
        """
        Class-aware wrapper for `::fz_new_ocr_device()`.
        	Create a device to OCR the text on the page.

        	Renders the page internally to a bitmap that is then OCRd. Text
        	is then forwarded onto the target device.

        	target: The target device to receive the OCRd text.

        	ctm: The transform to apply to the mediabox to get the size for
        	the rendered page image. Also used to calculate the resolution
        	for the page image. In general, this will be the same as the CTM
        	that you pass to fz_run_page (or fz_run_display_list) to feed
        	this device.

        	mediabox: The mediabox (in points). Combined with the CTM to get
        	the bounds of the pixmap used internally for the rendered page
        	image.

        	with_list: If with_list is false, then all non-text operations
        	are forwarded instantly to the target device. This results in
        	the target device seeing all NON-text operations, followed by
        	all the text operations (derived from OCR).

        	If with_list is true, then all the marking operations are
        	collated into a display list which is then replayed to the
        	target device at the end.

        	language: NULL (for "eng"), or a pointer to a string to describe
        	the languages/scripts that should be used for OCR (e.g.
        	"eng,ara").

        	datadir: NULL (for ""), or a pointer to a path string otherwise
        	provided to Tesseract in the TESSDATA_PREFIX environment variable.

        	progress: NULL, or function to be called periodically to indicate
        	progress. Return 0 to continue, or 1 to cancel. progress_arg is
        	returned as the void *. The int is a value between 0 and 100 to
        	indicate progress.

        	progress_arg: A void * value to be parrotted back to the progress
        	function.
        """
        return _mupdf.FzDevice_fz_new_ocr_device(self, ctm, mediabox, with_list, language, datadir, progress, progress_arg)

    def fz_pop_clip(self):
        """ Class-aware wrapper for `::fz_pop_clip()`."""
        return _mupdf.FzDevice_fz_pop_clip(self)

    def fz_render_flags(self, set, clear):
        """ Class-aware wrapper for `::fz_render_flags()`."""
        return _mupdf.FzDevice_fz_render_flags(self, set, clear)

    def fz_render_t3_glyph_direct(self, font, gid, trm, gstate, def_cs):
        """
        Class-aware wrapper for `::fz_render_t3_glyph_direct()`.
        	Nasty PDF interpreter specific hernia, required to allow the
        	interpreter to replay glyphs from a type3 font directly into
        	the target device.

        	This is only used in exceptional circumstances (such as type3
        	glyphs that inherit current graphics state, or nested type3
        	glyphs).
        """
        return _mupdf.FzDevice_fz_render_t3_glyph_direct(self, font, gid, trm, gstate, def_cs)

    def fz_set_default_colorspaces(self, default_cs):
        """ Class-aware wrapper for `::fz_set_default_colorspaces()`."""
        return _mupdf.FzDevice_fz_set_default_colorspaces(self, default_cs)

    def fz_stroke_path(self, path, stroke, ctm, colorspace, color, alpha, color_params):
        """ Class-aware wrapper for `::fz_stroke_path()`."""
        return _mupdf.FzDevice_fz_stroke_path(self, path, stroke, ctm, colorspace, color, alpha, color_params)

    def fz_stroke_text(self, text, stroke, ctm, colorspace, color, alpha, color_params):
        """ Class-aware wrapper for `::fz_stroke_text()`."""
        return _mupdf.FzDevice_fz_stroke_text(self, text, stroke, ctm, colorspace, color, alpha, color_params)

    def __init__(self, *args):
        """
        *Overload 1:*
         == Constructors.  Constructor using `fz_new_bbox_device()`.
        		Create a device to compute the bounding
        		box of all marks on a page.

        		The returned bounding box will be the union of all bounding
        		boxes of all objects on a page.


        |

        *Overload 2:*
         Constructor using `fz_new_device_of_size()`.
        		Devices are created by calls to device implementations, for
        		instance: foo_new_device(). These will be implemented by calling
        		fz_new_derived_device(ctx, foo_device) where foo_device is a
        		structure "derived from" fz_device, for instance
        		typedef struct { fz_device base;  ...extras...} foo_device;


        |

        *Overload 3:*
         Constructor using `fz_new_draw_device()`.
        		Create a device to draw on a pixmap.

        		dest: Target pixmap for the draw device. See fz_new_pixmap*
        		for how to obtain a pixmap. The pixmap is not cleared by the
        		draw device, see fz_clear_pixmap* for how to clear it prior to
        		calling fz_new_draw_device. Free the device by calling
        		fz_drop_device.

        		transform: Transform from user space in points to device space
        		in pixels.


        |

        *Overload 4:*
         Constructor using `fz_new_draw_device_with_bbox()`.
        		Create a device to draw on a pixmap.

        		dest: Target pixmap for the draw device. See fz_new_pixmap*
        		for how to obtain a pixmap. The pixmap is not cleared by the
        		draw device, see fz_clear_pixmap* for how to clear it prior to
        		calling fz_new_draw_device. Free the device by calling
        		fz_drop_device.

        		transform: Transform from user space in points to device space
        		in pixels.

        		clip: Bounding box to restrict any marking operations of the
        		draw device.


        |

        *Overload 5:*
         Constructor using `fz_new_draw_device_with_bbox_proof()`.
        		Create a device to draw on a pixmap.

        		dest: Target pixmap for the draw device. See fz_new_pixmap*
        		for how to obtain a pixmap. The pixmap is not cleared by the
        		draw device, see fz_clear_pixmap* for how to clear it prior to
        		calling fz_new_draw_device. Free the device by calling
        		fz_drop_device.

        		transform: Transform from user space in points to device space
        		in pixels.

        		clip: Bounding box to restrict any marking operations of the
        		draw device.

        		proof_cs: Color space to render to prior to mapping to color
        		space defined by pixmap.


        |

        *Overload 6:*
         Constructor using `fz_new_draw_device_with_options()`.
        		Create a new pixmap and draw device, using the specified options.

        		options: Options to configure the draw device, and choose the
        		resolution and colorspace.

        		mediabox: The bounds of the page in points.

        		pixmap: An out parameter containing the newly created pixmap.


        |

        *Overload 7:*
         Constructor using `fz_new_draw_device_with_proof()`.
        		Create a device to draw on a pixmap.

        		dest: Target pixmap for the draw device. See fz_new_pixmap*
        		for how to obtain a pixmap. The pixmap is not cleared by the
        		draw device, see fz_clear_pixmap* for how to clear it prior to
        		calling fz_new_draw_device. Free the device by calling
        		fz_drop_device.

        		transform: Transform from user space in points to device space
        		in pixels.

        		proof_cs: Intermediate color space to map though when mapping to
        		color space defined by pixmap.


        |

        *Overload 8:*
         Constructor using `fz_new_list_device()`.
        		Create a rendering device for a display list.

        		When the device is rendering a page it will populate the
        		display list with drawing commands (text, images, etc.). The
        		display list can later be reused to render a page many times
        		without having to re-interpret the page from the document file
        		for each rendering. Once the device is no longer needed, free
        		it with fz_drop_device.

        		list: A display list that the list device takes a reference to.


        |

        *Overload 9:*
         Constructor using `fz_new_ocr_device()`.
        		Create a device to OCR the text on the page.

        		Renders the page internally to a bitmap that is then OCRd. Text
        		is then forwarded onto the target device.

        		target: The target device to receive the OCRd text.

        		ctm: The transform to apply to the mediabox to get the size for
        		the rendered page image. Also used to calculate the resolution
        		for the page image. In general, this will be the same as the CTM
        		that you pass to fz_run_page (or fz_run_display_list) to feed
        		this device.

        		mediabox: The mediabox (in points). Combined with the CTM to get
        		the bounds of the pixmap used internally for the rendered page
        		image.

        		with_list: If with_list is false, then all non-text operations
        		are forwarded instantly to the target device. This results in
        		the target device seeing all NON-text operations, followed by
        		all the text operations (derived from OCR).

        		If with_list is true, then all the marking operations are
        		collated into a display list which is then replayed to the
        		target device at the end.

        		language: NULL (for "eng"), or a pointer to a string to describe
        		the languages/scripts that should be used for OCR (e.g.
        		"eng,ara").

        		datadir: NULL (for ""), or a pointer to a path string otherwise
        		provided to Tesseract in the TESSDATA_PREFIX environment variable.

        		progress: NULL, or function to be called periodically to indicate
        		progress. Return 0 to continue, or 1 to cancel. progress_arg is
        		returned as the void *. The int is a value between 0 and 100 to
        		indicate progress.

        		progress_arg: A void * value to be parrotted back to the progress
        		function.


        |

        *Overload 10:*
         Constructor using `fz_new_stext_device()`.
        		Create a device to extract the text on a page.

        		Gather the text on a page into blocks and lines.

        		The reading order is taken from the order the text is drawn in
        		the source file, so may not be accurate.

        		page: The text page to which content should be added. This will
        		usually be a newly created (empty) text page, but it can be one
        		containing data already (for example when merging multiple
        		pages, or watermarking).

        		options: Options to configure the stext device.


        |

        *Overload 11:*
         Constructor using `fz_new_svg_device()`.
        		Create a device that outputs (single page) SVG files to
        		the given output stream.

        		Equivalent to fz_new_svg_device_with_id passing id = NULL.


        |

        *Overload 12:*
         Constructor using `fz_new_svg_device_with_id()`.
        		Create a device that outputs (single page) SVG files to
        		the given output stream.

        		output: The output stream to send the constructed SVG page to.

        		page_width, page_height: The page dimensions to use (in points).

        		text_format: How to emit text. One of the following values:
        			FZ_SVG_TEXT_AS_TEXT: As <text> elements with possible
        			layout errors and mismatching fonts.
        			FZ_SVG_TEXT_AS_PATH: As <path> elements with exact
        			visual appearance.

        		reuse_images: Share image resources using <symbol> definitions.

        		id: ID parameter to keep generated IDs unique across SVG files.


        |

        *Overload 13:*
         Constructor using `fz_new_test_device()`.
        		Create a device to test for features.

        		Currently only tests for the presence of non-grayscale colors.

        		is_color: Possible values returned:
        			0: Definitely greyscale
        			1: Probably color (all colors were grey, but there
        			were images or shadings in a non grey colorspace).
        			2: Definitely color

        		threshold: The difference from grayscale that will be tolerated.
        		Typical values to use are either 0 (be exact) and 0.02 (allow an
        		imperceptible amount of slop).

        		options: A set of bitfield options, from the FZ_TEST_OPT set.

        		passthrough: A device to pass all calls through to, or NULL.
        		If set, then the test device can both test and pass through to
        		an underlying device (like, say, the display list device). This
        		means that a display list can be created and at the end we'll
        		know if it's colored or not.

        		In the absence of a passthrough device, the device will throw
        		an exception to stop page interpretation when color is found.


        |

        *Overload 14:*
         Constructor using `fz_new_trace_device()`.
        		Create a device to print a debug trace of all device calls.


        |

        *Overload 15:*
         Constructor using `pdf_new_pdf_device()`.

        |

        *Overload 16:*
         Copy constructor using `fz_keep_device()`.

        |

        *Overload 17:*
         Default constructor, sets `m_internal` to null.

        |

        *Overload 18:*
         Constructor using raw copy of pre-existing `::fz_device`.
        """
        _mupdf.FzDevice_swiginit(self, _mupdf.new_FzDevice(*args))
    __swig_destroy__ = _mupdf.delete_FzDevice

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzDevice_m_internal_value(self)
    m_internal = property(_mupdf.FzDevice_m_internal_get, _mupdf.FzDevice_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzDevice_s_num_instances_get, _mupdf.FzDevice_s_num_instances_set)