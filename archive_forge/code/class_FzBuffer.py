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
class FzBuffer(object):
    """
    Wrapper class for struct `fz_buffer`.
    fz_buffer is a wrapper around a dynamically allocated array of
    bytes.

    Buffers have a capacity (the number of bytes storage immediately
    available) and a current size.

    The contents of the structure are considered implementation
    details and are subject to change. Users should use the accessor
    functions in preference.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    @staticmethod
    def fz_new_buffer_from_copied_data(data, size):
        """
        Class-aware wrapper for `::fz_new_buffer_from_copied_data()`.
        	Create a new buffer containing a copy of the passed data.
        """
        return _mupdf.FzBuffer_fz_new_buffer_from_copied_data(data, size)

    @staticmethod
    def fz_new_buffer_from_image_as_pnm(image, color_params):
        """ Class-aware wrapper for `::fz_new_buffer_from_image_as_pnm()`."""
        return _mupdf.FzBuffer_fz_new_buffer_from_image_as_pnm(image, color_params)

    @staticmethod
    def fz_new_buffer_from_image_as_pam(image, color_params):
        """ Class-aware wrapper for `::fz_new_buffer_from_image_as_pam()`."""
        return _mupdf.FzBuffer_fz_new_buffer_from_image_as_pam(image, color_params)

    @staticmethod
    def fz_new_buffer_from_image_as_psd(image, color_params):
        """ Class-aware wrapper for `::fz_new_buffer_from_image_as_psd()`."""
        return _mupdf.FzBuffer_fz_new_buffer_from_image_as_psd(image, color_params)

    @staticmethod
    def fz_new_buffer_from_pixmap_as_pnm(pixmap, color_params):
        """ Class-aware wrapper for `::fz_new_buffer_from_pixmap_as_pnm()`."""
        return _mupdf.FzBuffer_fz_new_buffer_from_pixmap_as_pnm(pixmap, color_params)

    @staticmethod
    def fz_new_buffer_from_pixmap_as_pam(pixmap, color_params):
        """ Class-aware wrapper for `::fz_new_buffer_from_pixmap_as_pam()`."""
        return _mupdf.FzBuffer_fz_new_buffer_from_pixmap_as_pam(pixmap, color_params)

    @staticmethod
    def fz_new_buffer_from_pixmap_as_psd(pix, color_params):
        """ Class-aware wrapper for `::fz_new_buffer_from_pixmap_as_psd()`."""
        return _mupdf.FzBuffer_fz_new_buffer_from_pixmap_as_psd(pix, color_params)

    def fz_append_base64(self, data, size, newline):
        """
        Class-aware wrapper for `::fz_append_base64()`.
        	Write a base64 encoded data block, optionally with periodic newlines.
        """
        return _mupdf.FzBuffer_fz_append_base64(self, data, size, newline)

    def fz_append_base64_buffer(self, data, newline):
        """
        Class-aware wrapper for `::fz_append_base64_buffer()`.
        	Append a base64 encoded fz_buffer, optionally with periodic newlines.
        """
        return _mupdf.FzBuffer_fz_append_base64_buffer(self, data, newline)

    def fz_append_bits(self, value, count):
        """ Class-aware wrapper for `::fz_append_bits()`."""
        return _mupdf.FzBuffer_fz_append_bits(self, value, count)

    def fz_append_bits_pad(self):
        """ Class-aware wrapper for `::fz_append_bits_pad()`."""
        return _mupdf.FzBuffer_fz_append_bits_pad(self)

    def fz_append_buffer(self, source):
        """
        Class-aware wrapper for `::fz_append_buffer()`.
        	Append the contents of the source buffer onto the end of the
        	destination buffer, extending automatically as required.

        	Ownership of buffers does not change.
        """
        return _mupdf.FzBuffer_fz_append_buffer(self, source)

    def fz_append_byte(self, c):
        """ Class-aware wrapper for `::fz_append_byte()`."""
        return _mupdf.FzBuffer_fz_append_byte(self, c)

    def fz_append_data(self, data, len):
        """
        Class-aware wrapper for `::fz_append_data()`.
        	fz_append_*: Append data to a buffer.

        	The buffer will automatically grow as required.
        """
        return _mupdf.FzBuffer_fz_append_data(self, data, len)

    def fz_append_image_as_data_uri(self, image):
        """ Class-aware wrapper for `::fz_append_image_as_data_uri()`."""
        return _mupdf.FzBuffer_fz_append_image_as_data_uri(self, image)

    def fz_append_int16_be(self, x):
        """ Class-aware wrapper for `::fz_append_int16_be()`."""
        return _mupdf.FzBuffer_fz_append_int16_be(self, x)

    def fz_append_int16_le(self, x):
        """ Class-aware wrapper for `::fz_append_int16_le()`."""
        return _mupdf.FzBuffer_fz_append_int16_le(self, x)

    def fz_append_int32_be(self, x):
        """ Class-aware wrapper for `::fz_append_int32_be()`."""
        return _mupdf.FzBuffer_fz_append_int32_be(self, x)

    def fz_append_int32_le(self, x):
        """ Class-aware wrapper for `::fz_append_int32_le()`."""
        return _mupdf.FzBuffer_fz_append_int32_le(self, x)

    def fz_append_pdf_string(self, text):
        """
        Class-aware wrapper for `::fz_append_pdf_string()`.
        	fz_append_pdf_string: Append a string with PDF syntax quotes and
        	escapes.

        	The buffer will automatically grow as required.
        """
        return _mupdf.FzBuffer_fz_append_pdf_string(self, text)

    def fz_append_pixmap_as_data_uri(self, pixmap):
        """ Class-aware wrapper for `::fz_append_pixmap_as_data_uri()`."""
        return _mupdf.FzBuffer_fz_append_pixmap_as_data_uri(self, pixmap)

    def fz_append_rune(self, c):
        """ Class-aware wrapper for `::fz_append_rune()`."""
        return _mupdf.FzBuffer_fz_append_rune(self, c)

    def fz_append_string(self, data):
        """ Class-aware wrapper for `::fz_append_string()`."""
        return _mupdf.FzBuffer_fz_append_string(self, data)

    def fz_buffer_extract(self, data):
        """
        Class-aware wrapper for `::fz_buffer_extract()`.

        This method has out-params. Python/C# wrappers look like:
        	`fz_buffer_extract()` => `(size_t, unsigned char *data)`

        	Take ownership of buffer contents.

        	Performs the same task as fz_buffer_storage, but ownership of
        	the data buffer returns with this call. The buffer is left
        	empty.

        	Note: Bad things may happen if this is called on a buffer with
        	multiple references that is being used from multiple threads.

        	data: Pointer to place to retrieve data pointer.

        	Returns length of stream.
        """
        return _mupdf.FzBuffer_fz_buffer_extract(self, data)

    def fz_buffer_storage(self, datap):
        """
        Class-aware wrapper for `::fz_buffer_storage()`.

        This method has out-params. Python/C# wrappers look like:
        	`fz_buffer_storage()` => `(size_t, unsigned char *datap)`

        	Retrieve internal memory of buffer.

        	datap: Output parameter that will be pointed to the data.

        	Returns the current size of the data in bytes.
        """
        return _mupdf.FzBuffer_fz_buffer_storage(self, datap)

    def fz_clear_buffer(self):
        """
        Class-aware wrapper for `::fz_clear_buffer()`.
        	Empties the buffer. Storage is not freed, but is held ready
        	to be reused as the buffer is refilled.

        	Never throws exceptions.
        """
        return _mupdf.FzBuffer_fz_clear_buffer(self)

    def fz_clone_buffer(self):
        """
        Class-aware wrapper for `::fz_clone_buffer()`.
        	Make a new buffer, containing a copy of the data used in
        	the original.
        """
        return _mupdf.FzBuffer_fz_clone_buffer(self)

    def fz_grow_buffer(self):
        """
        Class-aware wrapper for `::fz_grow_buffer()`.
        	Make some space within a buffer (i.e. ensure that
        	capacity > size).
        """
        return _mupdf.FzBuffer_fz_grow_buffer(self)

    def fz_load_jbig2_globals(self):
        """
        Class-aware wrapper for `::fz_load_jbig2_globals()`.
        	Create a jbig2 globals record from a buffer.

        	Immutable once created.
        """
        return _mupdf.FzBuffer_fz_load_jbig2_globals(self)

    def fz_md5_buffer(self, digest):
        """
        Class-aware wrapper for `::fz_md5_buffer()`.
        	Create an MD5 digest from buffer contents.

        	Never throws exceptions.
        """
        return _mupdf.FzBuffer_fz_md5_buffer(self, digest)

    def fz_new_display_list_from_svg(self, base_uri, dir, w, h):
        """
        Class-aware wrapper for `::fz_new_display_list_from_svg()`.

        This method has out-params. Python/C# wrappers look like:
        	`fz_new_display_list_from_svg(const char *base_uri, ::fz_archive *dir)` => `(fz_display_list *, float w, float h)`

        	Parse an SVG document into a display-list.
        """
        return _mupdf.FzBuffer_fz_new_display_list_from_svg(self, base_uri, dir, w, h)

    def fz_new_image_from_buffer(self):
        """
        Class-aware wrapper for `::fz_new_image_from_buffer()`.
        	Create a new image from a
        	buffer of data, inferring its type from the format
        	of the data.
        """
        return _mupdf.FzBuffer_fz_new_image_from_buffer(self)

    def fz_new_image_from_svg(self, base_uri, dir):
        """
        Class-aware wrapper for `::fz_new_image_from_svg()`.
        	Create a scalable image from an SVG document.
        """
        return _mupdf.FzBuffer_fz_new_image_from_svg(self, base_uri, dir)

    def fz_open_buffer(self):
        """
        Class-aware wrapper for `::fz_open_buffer()`.
        	Open a buffer as a stream.

        	buf: The buffer to open. Ownership of the buffer is NOT passed
        	in (this function takes its own reference).

        	Returns pointer to newly created stream. May throw exceptions on
        	failure to allocate.
        """
        return _mupdf.FzBuffer_fz_open_buffer(self)

    def fz_parse_xml(self, preserve_white):
        """
        Class-aware wrapper for `::fz_parse_xml()`.
        	Parse the contents of buffer into a tree of xml nodes.

        	preserve_white: whether to keep or delete all-whitespace nodes.
        """
        return _mupdf.FzBuffer_fz_parse_xml(self, preserve_white)

    def fz_parse_xml_from_html5(self):
        """
        Class-aware wrapper for `::fz_parse_xml_from_html5()`.
        	Parse the contents of a buffer into a tree of XML nodes,
        	using the HTML5 parsing algorithm.
        """
        return _mupdf.FzBuffer_fz_parse_xml_from_html5(self)

    def fz_resize_buffer(self, capacity):
        """
        Class-aware wrapper for `::fz_resize_buffer()`.
        	Ensure that a buffer has a given capacity,
        	truncating data if required.

        	capacity: The desired capacity for the buffer. If the current
        	size of the buffer contents is smaller than capacity, it is
        	truncated.
        """
        return _mupdf.FzBuffer_fz_resize_buffer(self, capacity)

    def fz_save_buffer(self, filename):
        """
        Class-aware wrapper for `::fz_save_buffer()`.
        	Save the contents of a buffer to a file.
        """
        return _mupdf.FzBuffer_fz_save_buffer(self, filename)

    def fz_slice_buffer(self, start, end):
        """
        Class-aware wrapper for `::fz_slice_buffer()`.
        	Create a new buffer with a (subset of) the data from the buffer.

        	start: if >= 0, offset from start of buffer, if < 0 offset from end of buffer.

        	end: if >= 0, offset from start of buffer, if < 0 offset from end of buffer.

        """
        return _mupdf.FzBuffer_fz_slice_buffer(self, start, end)

    def fz_string_from_buffer(self):
        """
        Class-aware wrapper for `::fz_string_from_buffer()`.
        	Ensure that a buffer's data ends in a
        	0 byte, and return a pointer to it.
        """
        return _mupdf.FzBuffer_fz_string_from_buffer(self)

    def fz_subset_cff_for_gids(self, gids, num_gids, symbolic, cidfont):
        """
        Class-aware wrapper for `::fz_subset_cff_for_gids()`.

        This method has out-params. Python/C# wrappers look like:
        	`fz_subset_cff_for_gids(int num_gids, int symbolic, int cidfont)` => `(fz_buffer *, int gids)`
        """
        return _mupdf.FzBuffer_fz_subset_cff_for_gids(self, gids, num_gids, symbolic, cidfont)

    def fz_subset_ttf_for_gids(self, gids, num_gids, symbolic, cidfont):
        """
        Class-aware wrapper for `::fz_subset_ttf_for_gids()`.

        This method has out-params. Python/C# wrappers look like:
        	`fz_subset_ttf_for_gids(int num_gids, int symbolic, int cidfont)` => `(fz_buffer *, int gids)`
        """
        return _mupdf.FzBuffer_fz_subset_ttf_for_gids(self, gids, num_gids, symbolic, cidfont)

    def fz_terminate_buffer(self):
        """
        Class-aware wrapper for `::fz_terminate_buffer()`.
        	Zero-terminate buffer in order to use as a C string.

        	This byte is invisible and does not affect the length of the
        	buffer as returned by fz_buffer_storage. The zero byte is
        	written *after* the data, and subsequent writes will overwrite
        	the terminating byte.

        	Subsequent changes to the size of the buffer (such as by
        	fz_buffer_trim, fz_buffer_grow, fz_resize_buffer, etc) may
        	invalidate this.
        """
        return _mupdf.FzBuffer_fz_terminate_buffer(self)

    def fz_trim_buffer(self):
        """
        Class-aware wrapper for `::fz_trim_buffer()`.
        	Trim wasted capacity from a buffer by resizing internal memory.
        """
        return _mupdf.FzBuffer_fz_trim_buffer(self)

    def pdf_append_token(self, tok, lex):
        """ Class-aware wrapper for `::pdf_append_token()`."""
        return _mupdf.FzBuffer_pdf_append_token(self, tok, lex)

    def pdf_new_buffer_processor(self, ahxencode, newlines):
        """ Class-aware wrapper for `::pdf_new_buffer_processor()`."""
        return _mupdf.FzBuffer_pdf_new_buffer_processor(self, ahxencode, newlines)

    def __init__(self, *args):
        """
        *Overload 1:*
         == Constructors.  Constructor using `fz_new_buffer()`.

        |

        *Overload 2:*
         Constructor using `fz_new_buffer_from_base64()`.
        		Create a new buffer with data decoded from a base64 input string.


        |

        *Overload 3:*
         Constructor using `fz_new_buffer_from_data()`.
        		Create a new buffer with existing data.

        		data: Pointer to existing data.
        		size: Size of existing data.

        		Takes ownership of data. Does not make a copy. Calls fz_free on
        		the data when the buffer is deallocated. Do not use 'data' after
        		passing to this function.

        		Returns pointer to new buffer. Throws exception on allocation
        		failure.


        |

        *Overload 4:*
         Constructor using `fz_new_buffer_from_display_list()`.

        |

        *Overload 5:*
         Constructor using `fz_new_buffer_from_image_as_jpeg()`.

        |

        *Overload 6:*
         Constructor using `fz_new_buffer_from_image_as_jpx()`.

        |

        *Overload 7:*
         Constructor using `fz_new_buffer_from_image_as_png()`.
        		Reencode a given image as a PNG into a buffer.

        		Ownership of the buffer is returned.


        |

        *Overload 8:*
         Constructor using `fz_new_buffer_from_page()`.

        |

        *Overload 9:*
         Constructor using `fz_new_buffer_from_page_number()`.

        |

        *Overload 10:*
         Constructor using `fz_new_buffer_from_page_with_format()`.
        		Returns an fz_buffer containing a page after conversion to specified format.

        		page: The page to convert.
        		format, options: Passed to fz_new_document_writer_with_output() internally.
        		transform, cookie: Passed to fz_run_page() internally.


        |

        *Overload 11:*
         Constructor using `fz_new_buffer_from_pixmap_as_jpeg()`.

        |

        *Overload 12:*
         Constructor using `fz_new_buffer_from_pixmap_as_jpx()`.

        |

        *Overload 13:*
         Constructor using `fz_new_buffer_from_pixmap_as_png()`.
        		Reencode a given pixmap as a PNG into a buffer.

        		Ownership of the buffer is returned.


        |

        *Overload 14:*
         Constructor using `fz_new_buffer_from_shared_data()`.
        		Like fz_new_buffer, but does not take ownership.


        |

        *Overload 15:*
         Constructor using `fz_new_buffer_from_stext_page()`.
        		Convert structured text into plain text.


        |

        *Overload 16:*
         Constructor using `fz_read_file()`.
        		Read all the contents of a file into a buffer.


        |

        *Overload 17:*
         Copy constructor using `fz_keep_buffer()`.

        |

        *Overload 18:*
         Constructor using raw copy of pre-existing `::fz_buffer`.

        |

        *Overload 19:*
         Constructor using raw copy of pre-existing `::fz_buffer`.
        """
        _mupdf.FzBuffer_swiginit(self, _mupdf.new_FzBuffer(*args))
    __swig_destroy__ = _mupdf.delete_FzBuffer

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzBuffer_m_internal_value(self)
    m_internal = property(_mupdf.FzBuffer_m_internal_get, _mupdf.FzBuffer_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzBuffer_s_num_instances_get, _mupdf.FzBuffer_s_num_instances_set)