import ctypes
import io
import operator
import os
import sys
import weakref
from functools import reduce
from pathlib import Path
from tempfile import NamedTemporaryFile
from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontOptions, _encode_string
class PDFSurface(Surface):
    """Creates a PDF surface of the specified size in PostScript points
    to be written to ``target``.

    Note that the size of individual pages of the PDF output can vary.
    See :meth:`set_size`.

    The PDF surface backend recognizes the following MIME types
    for the data attached to a surface (see :meth:`~Surface.set_mime_data`)
    when it is used as a source pattern for drawing on this surface:
    ``image/jpeg`` and
    ``image/jp2``.
    If any of them is specified, the PDF backend emits an image
    with the content of MIME data
    (with the ``/DCTDecode`` or ``/JPXDecode`` filter, respectively)
    instead of a surface snapshot
    (with the ``/FlateDecode`` filter),
    which typically produces PDF with a smaller file size.

    ``target`` can be :obj:`None` to specify no output.
    This will generate a surface that may be queried and used as a source,
    without generating a temporary file.

    :param target:
        A filename,
        a binary mode :term:`file object` with a ``write`` method,
        or :obj:`None`.
    :param width_in_points:
        Width of the surface, in points (1 point == 1/72.0 inch)
    :param height_in_points:
        Height of the surface, in points (1 point == 1/72.0 inch)
    :type width_in_points: float
    :type height_in_points: float

    """

    def __init__(self, target, width_in_points, height_in_points):
        if hasattr(target, 'write') or target is None:
            write_func = _make_write_func(target)
            pointer = cairo.cairo_pdf_surface_create_for_stream(write_func, ffi.NULL, width_in_points, height_in_points)
        else:
            write_func = None
            pointer = cairo.cairo_pdf_surface_create(_encode_filename(target), width_in_points, height_in_points)
        Surface.__init__(self, pointer, target_keep_alive=write_func)

    def set_size(self, width_in_points, height_in_points):
        """Changes the size of a PDF surface
        for the current (and subsequent) pages.

        This method should only be called
        before any drawing operations have been performed on the current page.
        The simplest way to do this is to call this method
        immediately after creating the surface
        or immediately after completing a page with either
        :meth:`~Surface.show_page` or :meth:`~Surface.copy_page`.

        :param width_in_points:
            New width of the page, in points (1 point = 1/72.0 inch)
        :param height_in_points:
            New height of the page, in points (1 point = 1/72.0 inch)
        :type width_in_points: float
        :type height_in_points: float

        """
        cairo.cairo_pdf_surface_set_size(self._pointer, width_in_points, height_in_points)
        self._check_status()

    def add_outline(self, parent_id, utf8, link_attribs, flags=None):
        """Add an item to the document outline hierarchy.

        The outline has the ``utf8`` name and links to the location specified
        by ``link_attribs``. Link attributes have the same keys and values as
        the Link Tag, excluding the ``rect`` attribute. The item will be a
        child of the item with id ``parent_id``. Use ``PDF_OUTLINE_ROOT`` as
        the parent id of top level items.

        :param parent_id:
            the id of the parent item or ``PDF_OUTLINE_ROOT`` if this is a
            top level item.
        :param utf8: the name of the outline.
        :param link_attribs:
            the link attributes specifying where this outline links to.
        :param flags: outline item flags.

        :return: the id for the added item.

        *New in cairo 1.16.*

        *New in cairocffi 0.9.*

        """
        if flags is None:
            flags = 0
        value = cairo.cairo_pdf_surface_add_outline(self._pointer, parent_id, _encode_string(utf8), _encode_string(link_attribs), flags)
        self._check_status()
        return value

    def set_metadata(self, metadata, utf8):
        """Sets document metadata.

        The ``PDF_METADATA_CREATE_DATE`` and ``PDF_METADATA_MOD_DATE``
        values must be in ISO-8601 format: YYYY-MM-DDThh:mm:ss. An optional
        timezone of the form "[+/-]hh:mm" or "Z" for UTC time can be appended.
        All other metadata values can be any UTF-8 string.

        :param metadata: the metadata item to set.
        :param utf8: metadata value.

        *New in cairo 1.16.*

        *New in cairocffi 0.9.*

        """
        cairo.cairo_pdf_surface_set_metadata(self._pointer, metadata, _encode_string(utf8))
        self._check_status()

    def set_page_label(self, utf8):
        """Set page label for the current page.

        :param utf8: the page label.

        *New in cairo 1.16.*

        *New in cairocffi 0.9.*

        """
        cairo.cairo_pdf_surface_set_page_label(self._pointer, _encode_string(utf8))

    def set_thumbnail_size(self, width, height):
        """Set thumbnail image size for the current and all subsequent pages.

        Setting a width or height of 0 disables thumbnails for the current and
        subsequent pages.

        :param width: thumbnail width.
        :param height: thumbnail height.

        *New in cairo 1.16.*

        *New in cairocffi 0.9.*

        """
        cairo.cairo_pdf_surface_set_thumbnail_size(self._pointer, width, height)

    def restrict_to_version(self, version):
        """Restricts the generated PDF file to ``version``.

        See :meth:`get_versions` for a list of available version values
        that can be used here.

        This method should only be called
        before any drawing operations have been performed on the given surface.
        The simplest way to do this is to call this method
        immediately after creating the surface.

        :param version: A :ref:`PDF_VERSION` string.

        *New in cairo 1.10.*

        """
        cairo.cairo_pdf_surface_restrict_to_version(self._pointer, version)
        self._check_status()

    @staticmethod
    def get_versions():
        """Return the list of supported PDF versions.
        See :meth:`restrict_to_version`.

        :return: A list of :ref:`PDF_VERSION` strings.

        *New in cairo 1.10.*

        """
        versions = ffi.new('cairo_pdf_version_t const **')
        num_versions = ffi.new('int *')
        cairo.cairo_pdf_get_versions(versions, num_versions)
        versions = versions[0]
        return [versions[i] for i in range(num_versions[0])]

    @staticmethod
    def version_to_string(version):
        """Return the string representation of the given :ref:`PDF_VERSION`.
        See :meth:`get_versions` for a way to get
        the list of valid version ids.

        *New in cairo 1.10.*

        """
        c_string = cairo.cairo_pdf_version_to_string(version)
        if c_string == ffi.NULL:
            raise ValueError(version)
        return ffi.string(c_string).decode('ascii')