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
class FzOutput(object):
    """ Wrapper class for struct `fz_output`. Not copyable or assignable."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    Fixed_STDOUT = _mupdf.FzOutput_Fixed_STDOUT
    Fixed_STDERR = _mupdf.FzOutput_Fixed_STDERR
    Filter_HEX = _mupdf.FzOutput_Filter_HEX
    Filter_85 = _mupdf.FzOutput_Filter_85
    Filter_RLE = _mupdf.FzOutput_Filter_RLE

    def fz_close_output(self):
        """
        Class-aware wrapper for `::fz_close_output()`.
        	Flush pending output and close an output stream.
        """
        return _mupdf.FzOutput_fz_close_output(self)

    def fz_debug_store(self):
        """
        Class-aware wrapper for `::fz_debug_store()`.
        	Output debugging information for the current state of the store
        	to the given output channel.
        """
        return _mupdf.FzOutput_fz_debug_store(self)

    def fz_dump_glyph_cache_stats(self):
        """
        Class-aware wrapper for `::fz_dump_glyph_cache_stats()`.
        	Dump debug statistics for the glyph cache.
        """
        return _mupdf.FzOutput_fz_dump_glyph_cache_stats(self)

    def fz_flush_output(self):
        """
        Class-aware wrapper for `::fz_flush_output()`.
        	Flush unwritten data.
        """
        return _mupdf.FzOutput_fz_flush_output(self)

    def fz_new_svg_device(self, page_width, page_height, text_format, reuse_images):
        """
        Class-aware wrapper for `::fz_new_svg_device()`.
        	Create a device that outputs (single page) SVG files to
        	the given output stream.

        	Equivalent to fz_new_svg_device_with_id passing id = NULL.
        """
        return _mupdf.FzOutput_fz_new_svg_device(self, page_width, page_height, text_format, reuse_images)

    def fz_new_svg_device_with_id(self, page_width, page_height, text_format, reuse_images, id):
        """
        Class-aware wrapper for `::fz_new_svg_device_with_id()`.

        This method has out-params. Python/C# wrappers look like:
        	`fz_new_svg_device_with_id(float page_width, float page_height, int text_format, int reuse_images)` => `(fz_device *, int id)`

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
        """
        return _mupdf.FzOutput_fz_new_svg_device_with_id(self, page_width, page_height, text_format, reuse_images, id)

    def fz_new_trace_device(self):
        """
        Class-aware wrapper for `::fz_new_trace_device()`.
        	Create a device to print a debug trace of all device calls.
        """
        return _mupdf.FzOutput_fz_new_trace_device(self)

    def fz_new_xmltext_device(self):
        """
        Class-aware wrapper for `::fz_new_xmltext_device()`.
        	Create a device to output raw information.
        """
        return _mupdf.FzOutput_fz_new_xmltext_device(self)

    def fz_output_supports_stream(self):
        """
        Class-aware wrapper for `::fz_output_supports_stream()`.
        	Query whether a given fz_output supports fz_stream_from_output.
        """
        return _mupdf.FzOutput_fz_output_supports_stream(self)

    def fz_output_xml(self, item, level):
        """
        Class-aware wrapper for `::fz_output_xml()`.
        	Pretty-print an XML tree to given output.
        """
        return _mupdf.FzOutput_fz_output_xml(self, item, level)

    def fz_print_stext_header_as_html(self):
        """ Class-aware wrapper for `::fz_print_stext_header_as_html()`."""
        return _mupdf.FzOutput_fz_print_stext_header_as_html(self)

    def fz_print_stext_header_as_xhtml(self):
        """ Class-aware wrapper for `::fz_print_stext_header_as_xhtml()`."""
        return _mupdf.FzOutput_fz_print_stext_header_as_xhtml(self)

    def fz_print_stext_page_as_html(self, page, id):
        """
        Class-aware wrapper for `::fz_print_stext_page_as_html()`.
        	Output structured text to a file in HTML (visual) format.
        """
        return _mupdf.FzOutput_fz_print_stext_page_as_html(self, page, id)

    def fz_print_stext_page_as_json(self, page, scale):
        """
        Class-aware wrapper for `::fz_print_stext_page_as_json()`.
        	Output structured text to a file in JSON format.
        """
        return _mupdf.FzOutput_fz_print_stext_page_as_json(self, page, scale)

    def fz_print_stext_page_as_text(self, page):
        """
        Class-aware wrapper for `::fz_print_stext_page_as_text()`.
        	Output structured text to a file in plain-text UTF-8 format.
        """
        return _mupdf.FzOutput_fz_print_stext_page_as_text(self, page)

    def fz_print_stext_page_as_xhtml(self, page, id):
        """
        Class-aware wrapper for `::fz_print_stext_page_as_xhtml()`.
        	Output structured text to a file in XHTML (semantic) format.
        """
        return _mupdf.FzOutput_fz_print_stext_page_as_xhtml(self, page, id)

    def fz_print_stext_page_as_xml(self, page, id):
        """
        Class-aware wrapper for `::fz_print_stext_page_as_xml()`.
        	Output structured text to a file in XML format.
        """
        return _mupdf.FzOutput_fz_print_stext_page_as_xml(self, page, id)

    def fz_print_stext_trailer_as_html(self):
        """ Class-aware wrapper for `::fz_print_stext_trailer_as_html()`."""
        return _mupdf.FzOutput_fz_print_stext_trailer_as_html(self)

    def fz_print_stext_trailer_as_xhtml(self):
        """ Class-aware wrapper for `::fz_print_stext_trailer_as_xhtml()`."""
        return _mupdf.FzOutput_fz_print_stext_trailer_as_xhtml(self)

    def fz_seek_output(self, off, whence):
        """
        Class-aware wrapper for `::fz_seek_output()`.
        	Seek to the specified file position.
        	See fseek for arguments.

        	Throw an error on unseekable outputs.
        """
        return _mupdf.FzOutput_fz_seek_output(self, off, whence)

    def fz_set_stddbg(self):
        """
        Class-aware wrapper for `::fz_set_stddbg()`.
        	Set the output stream to be used for fz_stddbg. Set to NULL to
        	reset to default (stderr).
        """
        return _mupdf.FzOutput_fz_set_stddbg(self)

    def fz_stream_from_output(self):
        """
        Class-aware wrapper for `::fz_stream_from_output()`.
        	Obtain the fz_output in the form of a fz_stream.

        	This allows data to be read back from some forms of fz_output
        	object. When finished reading, the fz_stream should be released
        	by calling fz_drop_stream. Until the fz_stream is dropped, no
        	further operations should be performed on the fz_output object.
        """
        return _mupdf.FzOutput_fz_stream_from_output(self)

    def fz_tell_output(self):
        """
        Class-aware wrapper for `::fz_tell_output()`.
        	Return the current file position.

        	Throw an error on untellable outputs.
        """
        return _mupdf.FzOutput_fz_tell_output(self)

    def fz_truncate_output(self):
        """
        Class-aware wrapper for `::fz_truncate_output()`.
        	Truncate the output at the current position.

        	This allows output streams which have seeked back from the end
        	of their storage to be truncated at the current point.
        """
        return _mupdf.FzOutput_fz_truncate_output(self)

    def fz_write_base64(self, data, size, newline):
        """
        Class-aware wrapper for `::fz_write_base64()`.
        	Write a base64 encoded data block, optionally with periodic
        	newlines.
        """
        return _mupdf.FzOutput_fz_write_base64(self, data, size, newline)

    def fz_write_base64_buffer(self, data, newline):
        """
        Class-aware wrapper for `::fz_write_base64_buffer()`.
        	Write a base64 encoded fz_buffer, optionally with periodic
        	newlines.
        """
        return _mupdf.FzOutput_fz_write_base64_buffer(self, data, newline)

    def fz_write_bitmap_as_pbm(self, bitmap):
        """
        Class-aware wrapper for `::fz_write_bitmap_as_pbm()`.
        	Write a bitmap as a pbm.
        """
        return _mupdf.FzOutput_fz_write_bitmap_as_pbm(self, bitmap)

    def fz_write_bitmap_as_pcl(self, bitmap, pcl):
        """
        Class-aware wrapper for `::fz_write_bitmap_as_pcl()`.
        	Write a bitmap as mono PCL.
        """
        return _mupdf.FzOutput_fz_write_bitmap_as_pcl(self, bitmap, pcl)

    def fz_write_bitmap_as_pkm(self, bitmap):
        """
        Class-aware wrapper for `::fz_write_bitmap_as_pkm()`.
        	Write a CMYK bitmap as a pkm.
        """
        return _mupdf.FzOutput_fz_write_bitmap_as_pkm(self, bitmap)

    def fz_write_bitmap_as_pwg(self, bitmap, pwg):
        """
        Class-aware wrapper for `::fz_write_bitmap_as_pwg()`.
        	Write a bitmap as a PWG.
        """
        return _mupdf.FzOutput_fz_write_bitmap_as_pwg(self, bitmap, pwg)

    def fz_write_bitmap_as_pwg_page(self, bitmap, pwg):
        """
        Class-aware wrapper for `::fz_write_bitmap_as_pwg_page()`.
        	Write a bitmap as a PWG page.

        	Caller should provide a file header by calling
        	fz_write_pwg_file_header, but can then write several pages to
        	the same file.
        """
        return _mupdf.FzOutput_fz_write_bitmap_as_pwg_page(self, bitmap, pwg)

    def fz_write_bits(self, data, num_bits):
        """
        Class-aware wrapper for `::fz_write_bits()`.
        	Write num_bits of data to the end of the output stream, assumed to be packed
        	most significant bits first.
        """
        return _mupdf.FzOutput_fz_write_bits(self, data, num_bits)

    def fz_write_bits_sync(self):
        """
        Class-aware wrapper for `::fz_write_bits_sync()`.
        	Sync to byte boundary after writing bits.
        """
        return _mupdf.FzOutput_fz_write_bits_sync(self)

    def fz_write_buffer(self, data):
        """ Class-aware wrapper for `::fz_write_buffer()`."""
        return _mupdf.FzOutput_fz_write_buffer(self, data)

    def fz_write_byte(self, x):
        """ Class-aware wrapper for `::fz_write_byte()`."""
        return _mupdf.FzOutput_fz_write_byte(self, x)

    def fz_write_char(self, x):
        """ Class-aware wrapper for `::fz_write_char()`."""
        return _mupdf.FzOutput_fz_write_char(self, x)

    def fz_write_data(self, data, size):
        """
        Class-aware wrapper for `::fz_write_data()`.
        	Write data to output.

        	data: Pointer to data to write.
        	size: Size of data to write in bytes.
        """
        return _mupdf.FzOutput_fz_write_data(self, data, size)

    def fz_write_float_be(self, f):
        """ Class-aware wrapper for `::fz_write_float_be()`."""
        return _mupdf.FzOutput_fz_write_float_be(self, f)

    def fz_write_float_le(self, f):
        """ Class-aware wrapper for `::fz_write_float_le()`."""
        return _mupdf.FzOutput_fz_write_float_le(self, f)

    def fz_write_image_as_data_uri(self, image):
        """
        Class-aware wrapper for `::fz_write_image_as_data_uri()`.
        	Write image as a data URI (for HTML and SVG output).
        """
        return _mupdf.FzOutput_fz_write_image_as_data_uri(self, image)

    def fz_write_int16_be(self, x):
        """ Class-aware wrapper for `::fz_write_int16_be()`."""
        return _mupdf.FzOutput_fz_write_int16_be(self, x)

    def fz_write_int16_le(self, x):
        """ Class-aware wrapper for `::fz_write_int16_le()`."""
        return _mupdf.FzOutput_fz_write_int16_le(self, x)

    def fz_write_int32_be(self, x):
        """
        Class-aware wrapper for `::fz_write_int32_be()`.
        	Write different sized data to an output stream.
        """
        return _mupdf.FzOutput_fz_write_int32_be(self, x)

    def fz_write_int32_le(self, x):
        """ Class-aware wrapper for `::fz_write_int32_le()`."""
        return _mupdf.FzOutput_fz_write_int32_le(self, x)

    def fz_write_pixmap_as_data_uri(self, pixmap):
        """ Class-aware wrapper for `::fz_write_pixmap_as_data_uri()`."""
        return _mupdf.FzOutput_fz_write_pixmap_as_data_uri(self, pixmap)

    def fz_write_pixmap_as_jpeg(self, pix, quality, invert_cmyk):
        """
        Class-aware wrapper for `::fz_write_pixmap_as_jpeg()`.
        	Write a pixmap as a JPEG.
        """
        return _mupdf.FzOutput_fz_write_pixmap_as_jpeg(self, pix, quality, invert_cmyk)

    def fz_write_pixmap_as_jpx(self, pix, quality):
        """
        Class-aware wrapper for `::fz_write_pixmap_as_jpx()`.
        	Pixmap data as JP2K with no subsampling.

        	quality = 100 = lossless
        	otherwise for a factor of x compression use 100-x. (so 80 is 1:20 compression)
        """
        return _mupdf.FzOutput_fz_write_pixmap_as_jpx(self, pix, quality)

    def fz_write_pixmap_as_pam(self, pixmap):
        """
        Class-aware wrapper for `::fz_write_pixmap_as_pam()`.
        	Write a pixmap as a pnm (greyscale, rgb or cmyk, with or without
        	alpha).
        """
        return _mupdf.FzOutput_fz_write_pixmap_as_pam(self, pixmap)

    def fz_write_pixmap_as_pcl(self, pixmap, pcl):
        """
        Class-aware wrapper for `::fz_write_pixmap_as_pcl()`.
        	Write an (RGB) pixmap as color PCL.
        """
        return _mupdf.FzOutput_fz_write_pixmap_as_pcl(self, pixmap, pcl)

    def fz_write_pixmap_as_pclm(self, pixmap, options):
        """
        Class-aware wrapper for `::fz_write_pixmap_as_pclm()`.
        	Write a (Greyscale or RGB) pixmap as pclm.
        """
        return _mupdf.FzOutput_fz_write_pixmap_as_pclm(self, pixmap, options)

    def fz_write_pixmap_as_pdfocr(self, pixmap, options):
        """
        Class-aware wrapper for `::fz_write_pixmap_as_pdfocr()`.
        	Write a (Greyscale or RGB) pixmap as pdfocr.
        """
        return _mupdf.FzOutput_fz_write_pixmap_as_pdfocr(self, pixmap, options)

    def fz_write_pixmap_as_png(self, pixmap):
        """
        Class-aware wrapper for `::fz_write_pixmap_as_png()`.
        	Write a (Greyscale or RGB) pixmap as a png.
        """
        return _mupdf.FzOutput_fz_write_pixmap_as_png(self, pixmap)

    def fz_write_pixmap_as_pnm(self, pixmap):
        """
        Class-aware wrapper for `::fz_write_pixmap_as_pnm()`.
        	Write a pixmap as a pnm (greyscale or rgb, no alpha).
        """
        return _mupdf.FzOutput_fz_write_pixmap_as_pnm(self, pixmap)

    def fz_write_pixmap_as_ps(self, pixmap):
        """
        Class-aware wrapper for `::fz_write_pixmap_as_ps()`.
        	Write a (gray, rgb, or cmyk, no alpha) pixmap out as postscript.
        """
        return _mupdf.FzOutput_fz_write_pixmap_as_ps(self, pixmap)

    def fz_write_pixmap_as_psd(self, pixmap):
        """
        Class-aware wrapper for `::fz_write_pixmap_as_psd()`.
        	Write a pixmap as a PSD file.
        """
        return _mupdf.FzOutput_fz_write_pixmap_as_psd(self, pixmap)

    def fz_write_pixmap_as_pwg(self, pixmap, pwg):
        """
        Class-aware wrapper for `::fz_write_pixmap_as_pwg()`.
        	Write a pixmap as a PWG.
        """
        return _mupdf.FzOutput_fz_write_pixmap_as_pwg(self, pixmap, pwg)

    def fz_write_pixmap_as_pwg_page(self, pixmap, pwg):
        """
        Class-aware wrapper for `::fz_write_pixmap_as_pwg_page()`.
        	Write a pixmap as a PWG page.

        	Caller should provide a file header by calling
        	fz_write_pwg_file_header, but can then write several pages to
        	the same file.
        """
        return _mupdf.FzOutput_fz_write_pixmap_as_pwg_page(self, pixmap, pwg)

    def fz_write_ps_file_header(self):
        """
        Class-aware wrapper for `::fz_write_ps_file_header()`.
        	Write the file level header for ps band writer output.
        """
        return _mupdf.FzOutput_fz_write_ps_file_header(self)

    def fz_write_ps_file_trailer(self, pages):
        """
        Class-aware wrapper for `::fz_write_ps_file_trailer()`.
        	Write the file level trailer for ps band writer output.
        """
        return _mupdf.FzOutput_fz_write_ps_file_trailer(self, pages)

    def fz_write_pwg_file_header(self):
        """
        Class-aware wrapper for `::fz_write_pwg_file_header()`.
        	Output the file header to a pwg stream, ready for pages to follow it.
        """
        return _mupdf.FzOutput_fz_write_pwg_file_header(self)

    def fz_write_rune(self, rune):
        """
        Class-aware wrapper for `::fz_write_rune()`.
        	Write a UTF-8 encoded unicode character.
        """
        return _mupdf.FzOutput_fz_write_rune(self, rune)

    def fz_write_string(self, s):
        """
        Class-aware wrapper for `::fz_write_string()`.
        	Write a string. Does not write zero terminator.
        """
        return _mupdf.FzOutput_fz_write_string(self, s)

    def fz_write_uint16_be(self, x):
        """ Class-aware wrapper for `::fz_write_uint16_be()`."""
        return _mupdf.FzOutput_fz_write_uint16_be(self, x)

    def fz_write_uint16_le(self, x):
        """ Class-aware wrapper for `::fz_write_uint16_le()`."""
        return _mupdf.FzOutput_fz_write_uint16_le(self, x)

    def fz_write_uint32_be(self, x):
        """ Class-aware wrapper for `::fz_write_uint32_be()`."""
        return _mupdf.FzOutput_fz_write_uint32_be(self, x)

    def fz_write_uint32_le(self, x):
        """ Class-aware wrapper for `::fz_write_uint32_le()`."""
        return _mupdf.FzOutput_fz_write_uint32_le(self, x)

    def pdf_new_output_processor(self, ahxencode, newlines):
        """ Class-aware wrapper for `::pdf_new_output_processor()`."""
        return _mupdf.FzOutput_pdf_new_output_processor(self, ahxencode, newlines)

    def pdf_print_crypt(self, crypt):
        """ Class-aware wrapper for `::pdf_print_crypt()`."""
        return _mupdf.FzOutput_pdf_print_crypt(self, crypt)

    def pdf_print_encrypted_obj(self, obj, tight, ascii, crypt, num, gen, sep):
        """
        Class-aware wrapper for `::pdf_print_encrypted_obj()`.

        This method has out-params. Python/C# wrappers look like:
        	`pdf_print_encrypted_obj(::pdf_obj *obj, int tight, int ascii, ::pdf_crypt *crypt, int num, int gen)` => int sep
        """
        return _mupdf.FzOutput_pdf_print_encrypted_obj(self, obj, tight, ascii, crypt, num, gen, sep)

    def pdf_print_font(self, fontdesc):
        """ Class-aware wrapper for `::pdf_print_font()`."""
        return _mupdf.FzOutput_pdf_print_font(self, fontdesc)

    def pdf_print_obj(self, obj, tight, ascii):
        """ Class-aware wrapper for `::pdf_print_obj()`."""
        return _mupdf.FzOutput_pdf_print_obj(self, obj, tight, ascii)

    def pdf_write_digest(self, byte_range, field, digest_offset, digest_length, signer):
        """ Class-aware wrapper for `::pdf_write_digest()`."""
        return _mupdf.FzOutput_pdf_write_digest(self, byte_range, field, digest_offset, digest_length, signer)

    def __init__(self, *args):
        """
        *Overload 1:*
         == Constructors.  Constructor using `fz_new_arc4_output()`.

        |

        *Overload 2:*
         Constructor using `fz_new_deflate_output()`.

        |

        *Overload 3:*
         Constructor using `fz_new_log_for_module()`.
        		Internal function to actually do the opening of the logfile.

        		Caller should close/drop the output when finished with it.


        |

        *Overload 4:*
         Constructor using `fz_new_output()`.
        		Create a new output object with the given
        		internal state and function pointers.

        		state: Internal state (opaque to everything but implementation).

        		write: Function to output a given buffer.

        		close: Cleanup function to destroy state when output closed.
        		May permissibly be null.


        |

        *Overload 5:*
         Constructor using `fz_new_output_with_buffer()`.
        		Open an output stream that appends
        		to a buffer.

        		buf: The buffer to append to.


        |

        *Overload 6:*
         Constructor using `fz_new_output_with_path()`.
        		Open an output stream that writes to a
        		given path.

        		filename: The filename to write to (specified in UTF-8).

        		append: non-zero if we should append to the file, rather than
        		overwriting it.


        |

        *Overload 7:*
         Uses fz_stdout() or fz_stderr().

        |

        *Overload 8:*
         Calls one of: fz_new_asciihex_output(), fz_new_ascii85_output(), fz_new_rle_output().

        |

        *Overload 9:*
         Constructor using raw copy of pre-existing `::fz_output`.

        |

        *Overload 10:*
         Constructor using raw copy of pre-existing `::fz_output`.
        """
        _mupdf.FzOutput_swiginit(self, _mupdf.new_FzOutput(*args))
    __swig_destroy__ = _mupdf.delete_FzOutput

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzOutput_m_internal_value(self)
    m_internal = property(_mupdf.FzOutput_m_internal_get, _mupdf.FzOutput_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzOutput_s_num_instances_get, _mupdf.FzOutput_s_num_instances_set)