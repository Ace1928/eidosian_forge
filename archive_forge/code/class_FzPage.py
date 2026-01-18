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
class FzPage(object):
    """ Wrapper class for struct `fz_page`."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def fz_bound_page(self):
        """
        Class-aware wrapper for `::fz_bound_page()`.
        	Determine the size of a page at 72 dpi.
        """
        return _mupdf.FzPage_fz_bound_page(self)

    def fz_bound_page_box(self, box):
        """ Class-aware wrapper for `::fz_bound_page_box()`."""
        return _mupdf.FzPage_fz_bound_page_box(self, box)

    def fz_create_link(self, bbox, uri):
        """
        Class-aware wrapper for `::fz_create_link()`.
        	Create a new link on a page.
        """
        return _mupdf.FzPage_fz_create_link(self, bbox, uri)

    def fz_delete_link(self, link):
        """
        Class-aware wrapper for `::fz_delete_link()`.
        	Delete an existing link on a page.
        """
        return _mupdf.FzPage_fz_delete_link(self, link)

    def fz_load_links(self):
        """
        Class-aware wrapper for `::fz_load_links()`.
        	Load the list of links for a page.

        	Returns a linked list of all the links on the page, each with
        	its clickable region and link destination. Each link is
        	reference counted so drop and free the list of links by
        	calling fz_drop_link on the pointer return from fz_load_links.

        	page: Page obtained from fz_load_page.
        """
        return _mupdf.FzPage_fz_load_links(self)

    def fz_new_buffer_from_page(self, options):
        """ Class-aware wrapper for `::fz_new_buffer_from_page()`."""
        return _mupdf.FzPage_fz_new_buffer_from_page(self, options)

    def fz_new_buffer_from_page_with_format(self, format, options, transform, cookie):
        """
        Class-aware wrapper for `::fz_new_buffer_from_page_with_format()`.
        	Returns an fz_buffer containing a page after conversion to specified format.

        	page: The page to convert.
        	format, options: Passed to fz_new_document_writer_with_output() internally.
        	transform, cookie: Passed to fz_run_page() internally.
        """
        return _mupdf.FzPage_fz_new_buffer_from_page_with_format(self, format, options, transform, cookie)

    def fz_new_display_list_from_page(self):
        """
        Class-aware wrapper for `::fz_new_display_list_from_page()`.
        	Create a display list.

        	Ownership of the display list is returned to the caller.
        """
        return _mupdf.FzPage_fz_new_display_list_from_page(self)

    def fz_new_display_list_from_page_contents(self):
        """
        Class-aware wrapper for `::fz_new_display_list_from_page_contents()`.
        	Create a display list from page contents (no annotations).

        	Ownership of the display list is returned to the caller.
        """
        return _mupdf.FzPage_fz_new_display_list_from_page_contents(self)

    def fz_new_pixmap_from_page(self, ctm, cs, alpha):
        """ Class-aware wrapper for `::fz_new_pixmap_from_page()`."""
        return _mupdf.FzPage_fz_new_pixmap_from_page(self, ctm, cs, alpha)

    def fz_new_pixmap_from_page_contents(self, ctm, cs, alpha):
        """
        Class-aware wrapper for `::fz_new_pixmap_from_page_contents()`.
        	Render the page contents without annotations.

        	Ownership of the pixmap is returned to the caller.
        """
        return _mupdf.FzPage_fz_new_pixmap_from_page_contents(self, ctm, cs, alpha)

    def fz_new_pixmap_from_page_contents_with_separations(self, ctm, cs, seps, alpha):
        """ Class-aware wrapper for `::fz_new_pixmap_from_page_contents_with_separations()`."""
        return _mupdf.FzPage_fz_new_pixmap_from_page_contents_with_separations(self, ctm, cs, seps, alpha)

    def fz_new_pixmap_from_page_with_separations(self, ctm, cs, seps, alpha):
        """ Class-aware wrapper for `::fz_new_pixmap_from_page_with_separations()`."""
        return _mupdf.FzPage_fz_new_pixmap_from_page_with_separations(self, ctm, cs, seps, alpha)

    def fz_page_label(self, buf, size):
        """
        Class-aware wrapper for `::fz_page_label()`.
        	Get page label for a given page.
        """
        return _mupdf.FzPage_fz_page_label(self, buf, size)

    def fz_page_presentation(self, transition, duration):
        """
        Class-aware wrapper for `::fz_page_presentation()`.

        This method has out-params. Python/C# wrappers look like:
        	`fz_page_presentation(::fz_transition *transition)` => `(fz_transition *, float duration)`

        	Get the presentation details for a given page.

        	transition: A pointer to a transition struct to fill out.

        	duration: A pointer to a place to set the page duration in
        	seconds. Will be set to 0 if no transition is specified for the
        	page.

        	Returns: a pointer to the transition structure, or NULL if there
        	is no transition specified for the page.
        """
        return _mupdf.FzPage_fz_page_presentation(self, transition, duration)

    def fz_page_separations(self):
        """
        Class-aware wrapper for `::fz_page_separations()`.
        	Get the separations details for a page.
        	This will be NULL, unless the format specifically supports
        	separations (such as PDF files). May be NULL even
        	so, if there are no separations on a page.

        	Returns a reference that must be dropped.
        """
        return _mupdf.FzPage_fz_page_separations(self)

    def fz_page_uses_overprint(self):
        """
        Class-aware wrapper for `::fz_page_uses_overprint()`.
        	Query if a given page requires overprint.
        """
        return _mupdf.FzPage_fz_page_uses_overprint(self)

    def fz_run_page(self, dev, transform, cookie):
        """
        Class-aware wrapper for `::fz_run_page()`.
        	Run a page through a device.

        	page: Page obtained from fz_load_page.

        	dev: Device obtained from fz_new_*_device.

        	transform: Transform to apply to page. May include for example
        	scaling and rotation, see fz_scale, fz_rotate and fz_concat.
        	Set to fz_identity if no transformation is desired.

        	cookie: Communication mechanism between caller and library
        	rendering the page. Intended for multi-threaded applications,
        	while single-threaded applications set cookie to NULL. The
        	caller may abort an ongoing rendering of a page. Cookie also
        	communicates progress information back to the caller. The
        	fields inside cookie are continually updated while the page is
        	rendering.
        """
        return _mupdf.FzPage_fz_run_page(self, dev, transform, cookie)

    def fz_run_page_annots(self, dev, transform, cookie):
        """
        Class-aware wrapper for `::fz_run_page_annots()`.
        	Run the annotations on a page through a device.
        """
        return _mupdf.FzPage_fz_run_page_annots(self, dev, transform, cookie)

    def fz_run_page_contents(self, dev, transform, cookie):
        """
        Class-aware wrapper for `::fz_run_page_contents()`.
        	Run a page through a device. Just the main
        	page content, without the annotations, if any.

        	page: Page obtained from fz_load_page.

        	dev: Device obtained from fz_new_*_device.

        	transform: Transform to apply to page. May include for example
        	scaling and rotation, see fz_scale, fz_rotate and fz_concat.
        	Set to fz_identity if no transformation is desired.

        	cookie: Communication mechanism between caller and library
        	rendering the page. Intended for multi-threaded applications,
        	while single-threaded applications set cookie to NULL. The
        	caller may abort an ongoing rendering of a page. Cookie also
        	communicates progress information back to the caller. The
        	fields inside cookie are continually updated while the page is
        	rendering.
        """
        return _mupdf.FzPage_fz_run_page_contents(self, dev, transform, cookie)

    def fz_run_page_widgets(self, dev, transform, cookie):
        """
        Class-aware wrapper for `::fz_run_page_widgets()`.
        	Run the widgets on a page through a device.
        """
        return _mupdf.FzPage_fz_run_page_widgets(self, dev, transform, cookie)

    def fz_search_page(self, needle, hit_mark, hit_bbox, hit_max):
        """
        Class-aware wrapper for `::fz_search_page()`.

        This method has out-params. Python/C# wrappers look like:
        	`fz_search_page(const char *needle, ::fz_quad *hit_bbox, int hit_max)` => `(int, int hit_mark)`

        	Search for the 'needle' text on the page.
        	Record the hits in the hit_bbox array and return the number of
        	hits. Will stop looking once it has filled hit_max rectangles.
        """
        return _mupdf.FzPage_fz_search_page(self, needle, hit_mark, hit_bbox, hit_max)

    def pdf_page_from_fz_page(self):
        """ Class-aware wrapper for `::pdf_page_from_fz_page()`."""
        return _mupdf.FzPage_pdf_page_from_fz_page(self)

    def doc(self):
        """ Returns wrapper for .doc member."""
        return _mupdf.FzPage_doc(self)

    def __init__(self, *args):
        """
        *Overload 1:*
         == Constructors.  Constructor using `fz_load_chapter_page()`.
        		Load a page.

        		After fz_load_page is it possible to retrieve the size of the
        		page using fz_bound_page, or to render the page using
        		fz_run_page_*. Free the page by calling fz_drop_page.

        		chapter: chapter number, 0 is the first chapter of the document.
        		number: page number, 0 is the first page of the chapter.


        |

        *Overload 2:*
         Constructor using `fz_load_page()`.
        		Load a given page number from a document. This may be much less
        		efficient than loading by location (chapter+page) for some
        		document types.


        |

        *Overload 3:*
         Constructor using `fz_new_page_of_size()`.
        		Different document types will be implemented by deriving from
        		fz_page. This macro allocates such derived structures, and
        		initialises the base sections.


        |

        *Overload 4:*
         Return FzPage for pdfpage.m_internal.super.

        |

        *Overload 5:*
         Copy constructor using `fz_keep_page()`.

        |

        *Overload 6:*
         Default constructor, sets `m_internal` to null.

        |

        *Overload 7:*
         Constructor using raw copy of pre-existing `::fz_page`.
        """
        _mupdf.FzPage_swiginit(self, _mupdf.new_FzPage(*args))
    __swig_destroy__ = _mupdf.delete_FzPage

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzPage_m_internal_value(self)
    m_internal = property(_mupdf.FzPage_m_internal_get, _mupdf.FzPage_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzPage_s_num_instances_get, _mupdf.FzPage_s_num_instances_set)