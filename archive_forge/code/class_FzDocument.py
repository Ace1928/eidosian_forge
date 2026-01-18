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
class FzDocument(object):
    """ Wrapper class for struct `fz_document`."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def fz_authenticate_password(self, password):
        """
        Class-aware wrapper for `::fz_authenticate_password()`.
        	Test if the given password can decrypt the document.

        	password: The password string to be checked. Some document
        	specifications do not specify any particular text encoding, so
        	neither do we.

        	Returns 0 for failure to authenticate, non-zero for success.

        	For PDF documents, further information can be given by examining
        	the bits in the return code.

        		Bit 0 => No password required
        		Bit 1 => User password authenticated
        		Bit 2 => Owner password authenticated
        """
        return _mupdf.FzDocument_fz_authenticate_password(self, password)

    def fz_clamp_location(self, loc):
        """
        Class-aware wrapper for `::fz_clamp_location()`.
        	Clamps a location into valid chapter/page range. (First clamps
        	the chapter into range, then the page into range).
        """
        return _mupdf.FzDocument_fz_clamp_location(self, loc)

    def fz_count_chapter_pages(self, chapter):
        """
        Class-aware wrapper for `::fz_count_chapter_pages()`.
        	Return the number of pages in a chapter.
        	May return 0.
        """
        return _mupdf.FzDocument_fz_count_chapter_pages(self, chapter)

    def fz_count_chapters(self):
        """
        Class-aware wrapper for `::fz_count_chapters()`.
        	Return the number of chapters in the document.
        	At least 1.
        """
        return _mupdf.FzDocument_fz_count_chapters(self)

    def fz_count_pages(self):
        """
        Class-aware wrapper for `::fz_count_pages()`.
        	Return the number of pages in document

        	May return 0 for documents with no pages.
        """
        return _mupdf.FzDocument_fz_count_pages(self)

    def fz_document_output_intent(self):
        """
        Class-aware wrapper for `::fz_document_output_intent()`.
        	Find the output intent colorspace if the document has defined
        	one.

        	Returns a borrowed reference that should not be dropped, unless
        	it is kept first.
        """
        return _mupdf.FzDocument_fz_document_output_intent(self)

    def fz_document_supports_accelerator(self):
        """
        Class-aware wrapper for `::fz_document_supports_accelerator()`.
        	Query if the document supports the saving of accelerator data.
        """
        return _mupdf.FzDocument_fz_document_supports_accelerator(self)

    def fz_format_link_uri(self, dest):
        """
        Class-aware wrapper for `::fz_format_link_uri()`.
        	Format an internal link to a page number, location, and possible viewing parameters,
        	suitable for use with fz_create_link.

        	Returns a newly allocated string that the caller must free.
        """
        return _mupdf.FzDocument_fz_format_link_uri(self, dest)

    def fz_has_permission(self, p):
        """
        Class-aware wrapper for `::fz_has_permission()`.
        	Check permission flags on document.
        """
        return _mupdf.FzDocument_fz_has_permission(self, p)

    def fz_is_document_reflowable(self):
        """
        Class-aware wrapper for `::fz_is_document_reflowable()`.
        	Is the document reflowable.

        	Returns 1 to indicate reflowable documents, otherwise 0.
        """
        return _mupdf.FzDocument_fz_is_document_reflowable(self)

    def fz_last_page(self):
        """
        Class-aware wrapper for `::fz_last_page()`.
        	Function to get the location for the last page in the document.
        	Using this can be far more efficient in some cases than calling
        	fz_count_pages and using the page number.
        """
        return _mupdf.FzDocument_fz_last_page(self)

    def fz_layout_document(self, w, h, em):
        """
        Class-aware wrapper for `::fz_layout_document()`.
        	Layout reflowable document types.

        	w, h: Page size in points.
        	em: Default font size in points.
        """
        return _mupdf.FzDocument_fz_layout_document(self, w, h, em)

    def fz_load_chapter_page(self, chapter, page):
        """
        Class-aware wrapper for `::fz_load_chapter_page()`.
        	Load a page.

        	After fz_load_page is it possible to retrieve the size of the
        	page using fz_bound_page, or to render the page using
        	fz_run_page_*. Free the page by calling fz_drop_page.

        	chapter: chapter number, 0 is the first chapter of the document.
        	number: page number, 0 is the first page of the chapter.
        """
        return _mupdf.FzDocument_fz_load_chapter_page(self, chapter, page)

    def fz_load_outline(self):
        """
        Class-aware wrapper for `::fz_load_outline()`.
        	Load the hierarchical document outline.

        	Should be freed by fz_drop_outline.
        """
        return _mupdf.FzDocument_fz_load_outline(self)

    def fz_load_page(self, number):
        """
        Class-aware wrapper for `::fz_load_page()`.
        	Load a given page number from a document. This may be much less
        	efficient than loading by location (chapter+page) for some
        	document types.
        """
        return _mupdf.FzDocument_fz_load_page(self, number)

    def fz_location_from_page_number(self, number):
        """
        Class-aware wrapper for `::fz_location_from_page_number()`.
        	Converts from page number to chapter+page. This may cause many
        	chapters to be laid out in order to calculate the number of
        	pages within those chapters.
        """
        return _mupdf.FzDocument_fz_location_from_page_number(self, number)

    def fz_lookup_bookmark(self, mark):
        """
        Class-aware wrapper for `::fz_lookup_bookmark()`.
        	Find a bookmark and return its page number.
        """
        return _mupdf.FzDocument_fz_lookup_bookmark(self, mark)

    def fz_lookup_metadata(self, key, buf, size):
        """
        Class-aware wrapper for `::fz_lookup_metadata()`.
        	Retrieve document meta data strings.

        	doc: The document to query.

        	key: Which meta data key to retrieve...

        	Basic information:
        		'format'	-- Document format and version.
        		'encryption'	-- Description of the encryption used.

        	From the document information dictionary:
        		'info:Title'
        		'info:Author'
        		'info:Subject'
        		'info:Keywords'
        		'info:Creator'
        		'info:Producer'
        		'info:CreationDate'
        		'info:ModDate'

        	buf: The buffer to hold the results (a nul-terminated UTF-8
        	string).

        	size: Size of 'buf'.

        	Returns the number of bytes need to store the string plus terminator
        	(will be larger than 'size' if the output was truncated), or -1 if the
        	key is not recognized or found.
        """
        return _mupdf.FzDocument_fz_lookup_metadata(self, key, buf, size)

    def fz_lookup_metadata2(self, key):
        """
        Class-aware wrapper for `::fz_lookup_metadata2()`.
        C++ alternative to `fz_lookup_metadata()` that returns a `std::string`
        or calls `fz_throw()` if not found.
        """
        return _mupdf.FzDocument_fz_lookup_metadata2(self, key)

    def fz_needs_password(self):
        """
        Class-aware wrapper for `::fz_needs_password()`.
        	Check if a document is encrypted with a
        	non-blank password.
        """
        return _mupdf.FzDocument_fz_needs_password(self)

    def fz_new_buffer_from_page_number(self, number, options):
        """ Class-aware wrapper for `::fz_new_buffer_from_page_number()`."""
        return _mupdf.FzDocument_fz_new_buffer_from_page_number(self, number, options)

    def fz_new_display_list_from_page_number(self, number):
        """ Class-aware wrapper for `::fz_new_display_list_from_page_number()`."""
        return _mupdf.FzDocument_fz_new_display_list_from_page_number(self, number)

    def fz_new_pixmap_from_page_number(self, number, ctm, cs, alpha):
        """ Class-aware wrapper for `::fz_new_pixmap_from_page_number()`."""
        return _mupdf.FzDocument_fz_new_pixmap_from_page_number(self, number, ctm, cs, alpha)

    def fz_new_pixmap_from_page_number_with_separations(self, number, ctm, cs, seps, alpha):
        """ Class-aware wrapper for `::fz_new_pixmap_from_page_number_with_separations()`."""
        return _mupdf.FzDocument_fz_new_pixmap_from_page_number_with_separations(self, number, ctm, cs, seps, alpha)

    def fz_new_xhtml_document_from_document(self, *args):
        """
        *Overload 1:*
         Class-aware wrapper for `::fz_new_xhtml_document_from_document()`.
        		Use text extraction to convert the input document into XHTML,
        		then open the result as a new document that can be reflowed.


        |

        *Overload 2:*
         Class-aware wrapper for `::fz_new_xhtml_document_from_document()`.
        		Use text extraction to convert the input document into XHTML,
        		then open the result as a new document that can be reflowed.
        """
        return _mupdf.FzDocument_fz_new_xhtml_document_from_document(self, *args)

    def fz_next_page(self, loc):
        """
        Class-aware wrapper for `::fz_next_page()`.
        	Function to get the location of the next page (allowing for the
        	end of chapters etc). If at the end of the document, returns the
        	current location.
        """
        return _mupdf.FzDocument_fz_next_page(self, loc)

    def fz_open_reflowed_document(self, opts):
        """ Class-aware wrapper for `::fz_open_reflowed_document()`."""
        return _mupdf.FzDocument_fz_open_reflowed_document(self, opts)

    def fz_output_accelerator(self, accel):
        """
        Class-aware wrapper for `::fz_output_accelerator()`.
        	Output accelerator data for the document to a given output
        	stream.
        """
        return _mupdf.FzDocument_fz_output_accelerator(self, accel)

    def fz_page_number_from_location(self, loc):
        """
        Class-aware wrapper for `::fz_page_number_from_location()`.
        	Converts from chapter+page to page number. This may cause many
        	chapters to be laid out in order to calculate the number of
        	pages within those chapters.
        """
        return _mupdf.FzDocument_fz_page_number_from_location(self, loc)

    def fz_previous_page(self, loc):
        """
        Class-aware wrapper for `::fz_previous_page()`.
        	Function to get the location of the previous page (allowing for
        	the end of chapters etc). If already at the start of the
        	document, returns the current page.
        """
        return _mupdf.FzDocument_fz_previous_page(self, loc)

    def fz_process_opened_pages(self, process_openend_page, state):
        """
        Class-aware wrapper for `::fz_process_opened_pages()`.
        	Iterates over all opened pages of the document, calling the
        	provided callback for each page for processing. If the callback
        	returns non-NULL then the iteration stops and that value is returned
        	to the called of fz_process_opened_pages().

        	The state pointer provided to fz_process_opened_pages() is
        	passed on to the callback but is owned by the caller.

        	Returns the first non-NULL value returned by the callback,
        	or NULL if the callback returned NULL for all opened pages.
        """
        return _mupdf.FzDocument_fz_process_opened_pages(self, process_openend_page, state)

    def fz_resolve_link(self, uri, xp, yp):
        """
        Class-aware wrapper for `::fz_resolve_link()`.

        This method has out-params. Python/C# wrappers look like:
        	`fz_resolve_link(const char *uri)` => `(fz_location, float xp, float yp)`

        	Resolve an internal link to a page number.

        	xp, yp: Pointer to store coordinate of destination on the page.

        	Returns (-1,-1) if the URI cannot be resolved.
        """
        return _mupdf.FzDocument_fz_resolve_link(self, uri, xp, yp)

    def fz_run_document_structure(self, dev, cookie):
        """
        Class-aware wrapper for `::fz_run_document_structure()`.
        	Run the document structure through a device.

        	doc: Document in question.

        	dev: Device obtained from fz_new_*_device.

        	cookie: Communication mechanism between caller and library.
        	Intended for multi-threaded applications, while
        	single-threaded applications set cookie to NULL. The
        	caller may abort an ongoing rendering of a page. Cookie also
        	communicates progress information back to the caller. The
        	fields inside cookie are continually updated while the page is
        	rendering.
        """
        return _mupdf.FzDocument_fz_run_document_structure(self, dev, cookie)

    def fz_save_accelerator(self, accel):
        """
        Class-aware wrapper for `::fz_save_accelerator()`.
        	Save accelerator data for the document to a given file.
        """
        return _mupdf.FzDocument_fz_save_accelerator(self, accel)

    def fz_search_chapter_page_number(self, chapter, page, needle, hit_mark, hit_bbox, hit_max):
        """
        Class-aware wrapper for `::fz_search_chapter_page_number()`.

        This method has out-params. Python/C# wrappers look like:
        	`fz_search_chapter_page_number(int chapter, int page, const char *needle, ::fz_quad *hit_bbox, int hit_max)` => `(int, int hit_mark)`
        """
        return _mupdf.FzDocument_fz_search_chapter_page_number(self, chapter, page, needle, hit_mark, hit_bbox, hit_max)

    def fz_search_page2(self, number, needle, hit_max):
        """
        Class-aware wrapper for `::fz_search_page2()`.
        C++ alternative to fz_search_page() that returns information in a std::vector.
        """
        return _mupdf.FzDocument_fz_search_page2(self, number, needle, hit_max)

    def fz_search_page_number(self, number, needle, hit_mark, hit_bbox, hit_max):
        """
        Class-aware wrapper for `::fz_search_page_number()`.

        This method has out-params. Python/C# wrappers look like:
        	`fz_search_page_number(int number, const char *needle, ::fz_quad *hit_bbox, int hit_max)` => `(int, int hit_mark)`
        """
        return _mupdf.FzDocument_fz_search_page_number(self, number, needle, hit_mark, hit_bbox, hit_max)

    def fz_set_metadata(self, key, value):
        """ Class-aware wrapper for `::fz_set_metadata()`."""
        return _mupdf.FzDocument_fz_set_metadata(self, key, value)

    def pdf_count_pages_imp(self, chapter):
        """ Class-aware wrapper for `::pdf_count_pages_imp()`."""
        return _mupdf.FzDocument_pdf_count_pages_imp(self, chapter)

    def pdf_document_from_fz_document(self):
        """ Class-aware wrapper for `::pdf_document_from_fz_document()`."""
        return _mupdf.FzDocument_pdf_document_from_fz_document(self)

    def pdf_load_page_imp(self, chapter, number):
        """ Class-aware wrapper for `::pdf_load_page_imp()`."""
        return _mupdf.FzDocument_pdf_load_page_imp(self, chapter, number)

    def pdf_page_label_imp(self, chapter, page, buf, size):
        """ Class-aware wrapper for `::pdf_page_label_imp()`."""
        return _mupdf.FzDocument_pdf_page_label_imp(self, chapter, page, buf, size)

    def pdf_specifics(self):
        """ Class-aware wrapper for `::pdf_specifics()`."""
        return _mupdf.FzDocument_pdf_specifics(self)

    def __init__(self, *args):
        """
        *Overload 1:*
         == Constructors.  Constructor using `fz_open_accelerated_document()`.
        		Open a document file and read its basic structure so pages and
        		objects can be located. MuPDF will try to repair broken
        		documents (without actually changing the file contents).

        		The returned fz_document is used when calling most other
        		document related functions.

        		filename: a path to a file as it would be given to open(2).


        |

        *Overload 2:*
         Constructor using `fz_open_accelerated_document_with_stream()`.
        		Open a document using the specified stream object rather than
        		opening a file on disk.

        		magic: a string used to detect document type; either a file name
        		or mime-type.

        		stream: a stream of the document contents.

        		accel: NULL, or a stream of the 'accelerator' contents for this document.

        		NOTE: The caller retains ownership of 'stream' and 'accel' - the document will
        		take its own references if required.


        |

        *Overload 3:*
         Constructor using `fz_open_accelerated_document_with_stream_and_dir()`.
        		Open a document using the specified stream object rather than
        		opening a file on disk.

        		magic: a string used to detect document type; either a file name
        		or mime-type.

        		stream: a stream of the document contents.

        		accel: NULL, or a stream of the 'accelerator' contents for this document.

        		dir: NULL, or the 'directory context' for the stream contents.

        		NOTE: The caller retains ownership of 'stream', 'accel' and 'dir' - the document will
        		take its own references if required.


        |

        *Overload 4:*
         Constructor using `fz_open_document()`.
        		Open a document file and read its basic structure so pages and
        		objects can be located. MuPDF will try to repair broken
        		documents (without actually changing the file contents).

        		The returned fz_document is used when calling most other
        		document related functions.

        		filename: a path to a file as it would be given to open(2).


        |

        *Overload 5:*
         Constructor using `fz_open_document_with_buffer()`.
        		Open a document using a buffer rather than opening a file on disk.


        |

        *Overload 6:*
         Constructor using `fz_open_document_with_stream()`.
        		Open a document using the specified stream object rather than
        		opening a file on disk.

        		magic: a string used to detect document type; either a file name
        		or mime-type.

        		stream: a stream representing the contents of the document file.

        		NOTE: The caller retains ownership of 'stream' - the document will take its
        		own reference if required.


        |

        *Overload 7:*
         Constructor using `fz_open_document_with_stream_and_dir()`.
        		Open a document using the specified stream object rather than
        		opening a file on disk.

        		magic: a string used to detect document type; either a file name
        		or mime-type.

        		stream: a stream representing the contents of the document file.

        		dir: a 'directory context' for those filetypes that need it.

        		NOTE: The caller retains ownership of 'stream' and 'dir' - the document will
        		take its own references if required.


        |

        *Overload 8:*
         Returns a FzDocument for pdfdocument.m_internal.super.

        |

        *Overload 9:*
         Copy constructor using `fz_keep_document()`.

        |

        *Overload 10:*
         Constructor using raw copy of pre-existing `::fz_document`.

        |

        *Overload 11:*
         Constructor using raw copy of pre-existing `::fz_document`.
        """
        _mupdf.FzDocument_swiginit(self, _mupdf.new_FzDocument(*args))
    __swig_destroy__ = _mupdf.delete_FzDocument

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzDocument_m_internal_value(self)
    m_internal = property(_mupdf.FzDocument_m_internal_get, _mupdf.FzDocument_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzDocument_s_num_instances_get, _mupdf.FzDocument_s_num_instances_set)