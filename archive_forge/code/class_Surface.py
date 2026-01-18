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
class Surface(object):
    """The base class for all surface types.

    Should not be instantiated directly, but see :doc:`cffi_api`.
    An instance may be returned for cairo surface types
    that are not (yet) defined in cairocffi.

    A :class:`Surface` represents an image,
    either as the destination of a drawing operation
    or as source when drawing onto another surface.
    To draw to a :class:`Surface`,
    create a cairo :class:`Context` with the surface as the target.

    There are different sub-classes of :class:`Surface`
    for different drawing backends;
    for example, :class:`ImageSurface` is a bitmap image in memory.

    The initial contents of a surface after creation
    depend upon the manner of its creation.
    If cairo creates the surface and backing storage for the user,
    it will be initially cleared;
    for example, :class:`ImageSurface` and :meth:`create_similar`.
    Alternatively, if the user passes in a reference
    to some backing storage and asks cairo to wrap that in a :class:`Surface`,
    then the contents are not modified;
    for example, :class:`ImageSurface` with a ``data`` argument.

    """

    def __init__(self, pointer, target_keep_alive=None):
        self._pointer = ffi.gc(pointer, _keepref(cairo, cairo.cairo_surface_destroy))
        self._check_status()
        if hasattr(target_keep_alive, '__array_interface__'):
            is_empty = target_keep_alive.size == 0
        else:
            is_empty = target_keep_alive in (None, ffi.NULL)
        if not is_empty:
            keep_alive = KeepAlive(target_keep_alive)
            _check_status(cairo.cairo_surface_set_user_data(self._pointer, SURFACE_TARGET_KEY, *keep_alive.closure))
            keep_alive.save()

    def _check_status(self):
        _check_status(cairo.cairo_surface_status(self._pointer))

    @staticmethod
    def _from_pointer(pointer, incref):
        """Wrap an existing ``cairo_surface_t *`` cdata pointer.

        :type incref: bool
        :param incref:
            Whether increase the :ref:`reference count <refcounting>` now.
        :return:
            A new instance of :class:`Surface` or one of its sub-classes,
            depending on the surface’s type.

        """
        if pointer == ffi.NULL:
            raise ValueError('Null pointer')
        if incref:
            cairo.cairo_surface_reference(pointer)
        self = object.__new__(SURFACE_TYPE_TO_CLASS.get(cairo.cairo_surface_get_type(pointer), Surface))
        Surface.__init__(self, pointer)
        return self

    def create_similar(self, content, width, height):
        """Create a new surface that is as compatible as possible
        for uploading to and the use in conjunction with this surface.
        For example the new surface will have the same fallback resolution
        and :class:`FontOptions`.
        Generally, the new surface will also use the same backend as other,
        unless that is not possible for some reason.

        Initially the surface contents are all 0
        (transparent if contents have transparency, black otherwise.)

        Use :meth:`create_similar_image` if you need an image surface
        which can be painted quickly to the target surface.

        :param content: the :ref:`CONTENT` string for the new surface.
        :param width: width of the new surface (in device-space units)
        :param height: height of the new surface (in device-space units)
        :type content: str
        :type width: int
        :type height: int
        :returns: A new instance of :class:`Surface` or one of its subclasses.

        """
        return Surface._from_pointer(cairo.cairo_surface_create_similar(self._pointer, content, width, height), incref=False)

    def create_similar_image(self, content, width, height):
        """
        Create a new image surface that is as compatible as possible
        for uploading to and the use in conjunction with this surface.
        However, this surface can still be used like any normal image surface.

        Initially the surface contents are all 0
        (transparent if contents have transparency, black otherwise.)

        Use :meth:`create_similar` if you don't need an image surface.

        :param format: the :ref:`FORMAT` string for the new surface
        :param width: width of the new surface, (in device-space units)
        :param height: height of the new surface (in device-space units)
        :type format: str
        :type width: int
        :type height: int
        :returns: A new :class:`ImageSurface` instance.

        """
        return Surface._from_pointer(cairo.cairo_surface_create_similar_image(self._pointer, content, width, height), incref=False)

    def create_for_rectangle(self, x, y, width, height):
        """
        Create a new surface that is a rectangle within this surface.
        All operations drawn to this surface are then clipped and translated
        onto the target surface.
        Nothing drawn via this sub-surface outside of its bounds
        is drawn onto the target surface,
        making this a useful method for passing constrained child surfaces
        to library routines that draw directly onto the parent surface,
        i.e. with no further backend allocations,
        double buffering or copies.

        .. note::

            As of cairo 1.12,
            the semantics of subsurfaces have not been finalized yet
            unless the rectangle is in full device units,
            is contained within the extents of the target surface,
            and the target or subsurface's device transforms are not changed.

        :param x:
            The x-origin of the sub-surface
            from the top-left of the target surface (in device-space units)
        :param y:
            The y-origin of the sub-surface
            from the top-left of the target surface (in device-space units)
        :param width:
            Width of the sub-surface (in device-space units)
        :param height:
            Height of the sub-surface (in device-space units)
        :type x: float
        :type y: float
        :type width: float
        :type height: float
        :returns:
            A new :class:`Surface` object.

        *New in cairo 1.10.*

        """
        return Surface._from_pointer(cairo.cairo_surface_create_for_rectangle(self._pointer, x, y, width, height), incref=False)

    def get_content(self):
        """Returns the :ref:`CONTENT` string of this surface,
        which indicates whether the surface contains color
        and/or alpha information.

        """
        return cairo.cairo_surface_get_content(self._pointer)

    def has_show_text_glyphs(self):
        """Returns whether the surface supports sophisticated
        :meth:`Context.show_text_glyphs` operations.
        That is, whether it actually uses the text and cluster data
        provided to a :meth:`Context.show_text_glyphs` call.

        .. note::

            Even if this method returns :obj:`False`,
            :meth:`Context.show_text_glyphs` operation targeted at surface
            will still succeed.
            It just will act like a :meth:`Context.show_glyphs` operation.
            Users can use this method to avoid computing UTF-8 text
            and cluster mapping if the target surface does not use it.

        """
        return bool(cairo.cairo_surface_has_show_text_glyphs(self._pointer))

    def set_device_offset(self, x_offset, y_offset):
        """ Sets an offset that is added to the device coordinates
        determined by the CTM when drawing to surface.
        One use case for this method is
        when we want to create a :class:`Surface` that redirects drawing
        for a portion of an onscreen surface
        to an offscreen surface in a way that is
        completely invisible to the user of the cairo API.
        Setting a transformation via :meth:`Context.translate`
        isn't sufficient to do this,
        since methods like :meth:`Context.device_to_user`
        will expose the hidden offset.

        Note that the offset affects drawing to the surface
        as well as using the surface in a source pattern.

        :param x_offset:
            The offset in the X direction, in device units
        :param y_offset:
            The offset in the Y direction, in device units

        """
        cairo.cairo_surface_set_device_offset(self._pointer, x_offset, y_offset)
        self._check_status()

    def get_device_offset(self):
        """Returns the previous device offset set by :meth:`set_device_offset`.

        :returns: ``(x_offset, y_offset)``

        """
        offsets = ffi.new('double[2]')
        cairo.cairo_surface_get_device_offset(self._pointer, offsets + 0, offsets + 1)
        return tuple(offsets)

    def set_fallback_resolution(self, x_pixels_per_inch, y_pixels_per_inch):
        """
        Set the horizontal and vertical resolution for image fallbacks.

        When certain operations aren't supported natively by a backend,
        cairo will fallback by rendering operations to an image
        and then overlaying that image onto the output.
        For backends that are natively vector-oriented,
        this method can be used to set the resolution
        used for these image fallbacks,
        (larger values will result in more detailed images,
        but also larger file sizes).

        Some examples of natively vector-oriented backends are
        the ps, pdf, and svg backends.

        For backends that are natively raster-oriented,
        image fallbacks are still possible,
        but they are always performed at the native device resolution.
        So this method has no effect on those backends.

        .. note::

            The fallback resolution only takes effect
            at the time of completing a page
            (with :meth:`show_page` or :meth:`copy_page`)
            so there is currently no way to have
            more than one fallback resolution in effect on a single page.

        The default fallback resoultion is
        300 pixels per inch in both dimensions.

        :param x_pixels_per_inch: horizontal resolution in pixels per inch
        :type x_pixels_per_inch: float
        :param y_pixels_per_inch: vertical resolution in pixels per inch
        :type y_pixels_per_inch: float

        """
        cairo.cairo_surface_set_fallback_resolution(self._pointer, x_pixels_per_inch, y_pixels_per_inch)
        self._check_status()

    def get_fallback_resolution(self):
        """Returns the previous fallback resolution
        set by :meth:`set_fallback_resolution`,
        or default fallback resolution if never set.

        :returns: ``(x_pixels_per_inch, y_pixels_per_inch)``

        """
        ppi = ffi.new('double[2]')
        cairo.cairo_surface_get_fallback_resolution(self._pointer, ppi + 0, ppi + 1)
        return tuple(ppi)

    def get_font_options(self):
        """Retrieves the default font rendering options for the surface.

        This allows display surfaces to report the correct subpixel order
        for rendering on them,
        print surfaces to disable hinting of metrics and so forth.
        The result can then be used with :class:`ScaledFont`.

        :returns: A new :class:`FontOptions` object.

        """
        font_options = FontOptions()
        cairo.cairo_surface_get_font_options(self._pointer, font_options._pointer)
        return font_options

    def set_device_scale(self, x_scale, y_scale):
        """Sets a scale that is multiplied to the device coordinates determined
        by the CTM when drawing to surface.

        One common use for this is to render to very high resolution display
        devices at a scale factor, so that code that assumes 1 pixel will be a
        certain size will still work.  Setting a transformation via
        cairo_translate() isn't sufficient to do this, since functions like
        cairo_device_to_user() will expose the hidden scale.

        Note that the scale affects drawing to the surface as well as using the
        surface in a source pattern.

        :param x_scale: the scale in the X direction, in device units.
        :param y_scale: the scale in the Y direction, in device units.

        *New in cairo 1.14.*

        *New in cairocffi 0.9.*

        """
        cairo.cairo_surface_set_device_scale(self._pointer, x_scale, y_scale)
        self._check_status()

    def get_device_scale(self):
        """Returns the previous device offset set by :meth:`set_device_scale`.

        *New in cairo 1.14.*

        *New in cairocffi 0.9.*

        """
        size = ffi.new('double[2]')
        cairo.cairo_surface_get_device_scale(self._pointer, size + 0, size + 1)
        return tuple(size)

    def set_mime_data(self, mime_type, data):
        """
        Attach an image in the format ``mime_type`` to this surface.

        To remove the data from a surface,
        call this method with same mime type and :obj:`None` for data.

        The attached image (or filename) data can later
        be used by backends which support it
        (currently: PDF, PS, SVG and Win32 Printing surfaces)
        to emit this data instead of making a snapshot of the surface.
        This approach tends to be faster
        and requires less memory and disk space.

        The recognized MIME types are the following:

        ``"image/png"``
            The Portable Network Graphics image file format (ISO/IEC 15948).
        ``"image/jpeg"``
            The Joint Photographic Experts Group (JPEG)
            image coding standard (ISO/IEC 10918-1).
        ``"image/jp2"``
            The Joint Photographic Experts Group (JPEG) 2000
            image coding standard (ISO/IEC 15444-1).
        ``"text/x-uri"``
            URL for an image file (unofficial MIME type).

        See corresponding backend surface docs
        for details about which MIME types it can handle.
        Caution: the associated MIME data will be discarded
        if you draw on the surface afterwards.
        Use this method with care.

        :param str mime_type: The MIME type of the image data.
        :param bytes data: The image data to attach to the surface.

        *New in cairo 1.10.*

        """
        mime_type = ffi.new('char[]', mime_type.encode('utf8'))
        if data is None:
            _check_status(cairo.cairo_surface_set_mime_data(self._pointer, mime_type, ffi.NULL, 0, ffi.NULL, ffi.NULL))
        else:
            length = len(data)
            data = ffi.new('unsigned char[]', data)
            keep_alive = KeepAlive(data, mime_type)
            _check_status(cairo.cairo_surface_set_mime_data(self._pointer, mime_type, data, length, *keep_alive.closure))
            keep_alive.save()

    def get_mime_data(self, mime_type):
        """Return mime data previously attached to surface
        using the specified mime type.

        :param str mime_type: The MIME type of the image data.
        :returns:
            A CFFI buffer object, or :obj:`None`
            if no data has been attached with the given mime type.

        *New in cairo 1.10.*

        """
        buffer_address = ffi.new('unsigned char **')
        buffer_length = ffi.new('unsigned long *')
        mime_type = ffi.new('char[]', mime_type.encode('utf8'))
        cairo.cairo_surface_get_mime_data(self._pointer, mime_type, buffer_address, buffer_length)
        return ffi.buffer(buffer_address[0], buffer_length[0]) if buffer_address[0] != ffi.NULL else None

    def supports_mime_type(self, mime_type):
        """Return whether surface supports ``mime_type``.

        :param str mime_type: The MIME type of the image data.

        *New in cairo 1.12.*

        """
        mime_type = ffi.new('char[]', mime_type.encode('utf8'))
        return bool(cairo.cairo_surface_supports_mime_type(self._pointer, mime_type))

    def mark_dirty(self):
        """Tells cairo that drawing has been done to surface
        using means other than cairo,
        and that cairo should reread any cached areas.
        Note that you must call :meth:`flush` before doing such drawing.

        """
        cairo.cairo_surface_mark_dirty(self._pointer)
        self._check_status()

    def mark_dirty_rectangle(self, x, y, width, height):
        """
        Like :meth:`mark_dirty`,
        but drawing has been done only to the specified rectangle,
        so that cairo can retain cached contents
        for other parts of the surface.

        Any cached clip set on the surface will be reset by this method,
        to make sure that future cairo calls have the clip set
        that they expect.

        :param x: X coordinate of dirty rectangle.
        :param y: Y coordinate of dirty rectangle.
        :param width: Width of dirty rectangle.
        :param height: Height of dirty rectangle.
        :type x: float
        :type y: float
        :type width: float
        :type height: float

        """
        cairo.cairo_surface_mark_dirty_rectangle(self._pointer, x, y, width, height)
        self._check_status()

    def show_page(self):
        """Emits and clears the current page
        for backends that support multiple pages.
        Use :meth:`copy_page` if you don't want to clear the page.

        :meth:`Context.show_page` is a convenience method for this.

        """
        cairo.cairo_surface_show_page(self._pointer)
        self._check_status()

    def copy_page(self):
        """Emits the current page for backends that support multiple pages,
        but doesn't clear it,
        so that the contents of the current page will be retained
        for the next page.

        Use :meth:`show_page` if you want to get an empty page
        after the emission.

        """
        cairo.cairo_surface_copy_page(self._pointer)
        self._check_status()

    def flush(self):
        """Do any pending drawing for the surface
        and also restore any temporary modifications
        cairo has made to the surface's state.
        This method must be called before switching
        from drawing on the surface with cairo
        to drawing on it directly with native APIs.
        If the surface doesn't support direct access,
        then this method does nothing.

        """
        cairo.cairo_surface_flush(self._pointer)
        self._check_status()

    def finish(self):
        """This method finishes the surface
        and drops all references to external resources.
        For example, for the Xlib backend it means that
        cairo will no longer access the drawable, which can be freed.
        After calling :meth:`finish` the only valid operations on a surface
        are getting and setting user data, flushing and finishing it.
        Further drawing to the surface will not affect the surface
        but will instead trigger a :class:`CairoError`
        with a ``SURFACE_FINISHED`` status.

        When the surface is garbage-collected, cairo will call :meth:`finish()`
        if it hasn't been called already,
        before freeing the resources associated with the surface.

        """
        cairo.cairo_surface_finish(self._pointer)
        self._check_status()

    def write_to_png(self, target=None):
        """Writes the contents of surface as a PNG image.

        :param target:
            A filename,
            a binary mode :term:`file object` with a `write` method,
            or :obj:`None`.
        :returns:
            If ``target`` is :obj:`None`,
            return the PNG contents as a byte string.

        """
        return_bytes = target is None
        if return_bytes:
            target = io.BytesIO()
        if hasattr(target, 'write'):
            try:
                write_func = _make_write_func(target)
                _check_status(cairo.cairo_surface_write_to_png_stream(self._pointer, write_func, ffi.NULL))
            except (SystemError, MemoryError):
                if hasattr(target, 'name'):
                    _check_status(cairo.cairo_surface_write_to_png(self._pointer, _encode_filename(target.name)))
                else:
                    with NamedTemporaryFile('wb', delete=False) as fd:
                        filename = fd.name
                        _check_status(cairo.cairo_surface_write_to_png(self._pointer, _encode_filename(filename)))
                    png_file = Path(filename)
                    target.write(png_file.read_bytes())
                    png_file.unlink()
        else:
            _check_status(cairo.cairo_surface_write_to_png(self._pointer, _encode_filename(target)))
        if return_bytes:
            return target.getvalue()