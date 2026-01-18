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
class ImageSurface(Surface):
    """Creates an image surface of the specified format and dimensions.

    If ``data`` is not :obj:`None`
    its initial contents will be used as the initial image contents;
    you must explicitly clear the buffer,
    using, for example, :meth:`Context.rectangle` and :meth:`Context.fill`
    if you want it cleared.

    .. note::

        Currently only :class:`array.array` buffers are supported on PyPy.

    Otherwise, the surface contents are all initially 0.
    (Specifically, within each pixel, each color or alpha channel
    belonging to format will be 0.
    The contents of bits within a pixel,
    but not belonging to the given format are undefined).

    :param format: :ref:`FORMAT` string for the surface to create.
    :param width: Width of the surface, in pixels.
    :param height: Height of the surface, in pixels.
    :param data:
        Buffer supplied in which to write contents,
        or :obj:`None` to create a new buffer.
    :param stride:
        The number of bytes between the start of rows
        in the buffer as allocated.
        This value should always be computed by :meth:`format_stride_for_width`
        before allocating the data buffer.
        If omitted but ``data`` is given,
        :meth:`format_stride_for_width` is used.
    :type format: str
    :type width: int
    :type height: int
    :type stride: int

    """

    def __init__(self, format, width, height, data=None, stride=None):
        if data is None:
            pointer = cairo.cairo_image_surface_create(format, width, height)
        else:
            if stride is None:
                stride = self.format_stride_for_width(format, width)
            address, length = from_buffer(data)
            if length < stride * height:
                raise ValueError('Got a %d bytes buffer, needs at least %d.' % (length, stride * height))
            pointer = cairo.cairo_image_surface_create_for_data(ffi.cast('unsigned char*', address), format, width, height, stride)
        Surface.__init__(self, pointer, target_keep_alive=data)

    @classmethod
    def create_for_data(cls, data, format, width, height, stride=None):
        """Same as ``ImageSurface(format, width, height, data, stride)``.
        Exists for compatibility with pycairo.

        """
        return cls(format, width, height, data, stride)

    @staticmethod
    def format_stride_for_width(format, width):
        """
        This method provides a stride value (byte offset between rows)
        that will respect all alignment requirements
        of the accelerated image-rendering code within cairo.
        Typical usage will be of the form::

            from cairocffi import ImageSurface
            stride = ImageSurface.format_stride_for_width(format, width)
            data = bytearray(stride * height)
            surface = ImageSurface(format, width, height, data, stride)

        :param format: A :ref:`FORMAT` string.
        :param width: The desired width of the surface, in pixels.
        :type format: str
        :type width: int
        :returns:
            The appropriate stride to use given the desired format and width,
            or -1 if either the format is invalid or the width too large.

        """
        return cairo.cairo_format_stride_for_width(format, width)

    @classmethod
    def create_from_png(cls, source):
        """Decode a PNG file into a new image surface.

        :param source:
            A filename or
            a binary mode :term:`file object` with a ``read`` method.
            If you already have a byte string in memory,
            use :class:`io.BytesIO`.
        :returns: A new :class:`ImageSurface` instance.

        """
        if hasattr(source, 'read'):
            read_func = _make_read_func(source)
            pointer = cairo.cairo_image_surface_create_from_png_stream(read_func, ffi.NULL)
        else:
            pointer = cairo.cairo_image_surface_create_from_png(_encode_filename(source))
        self = object.__new__(cls)
        Surface.__init__(self, pointer)
        return self

    def get_data(self):
        """Return the buffer pointing to the image’s pixel data,
        encoded according to the surface’s :ref:`FORMAT` string.

        A call to :meth:`~Surface.flush` is required before accessing the pixel
        data to ensure that all pending drawing operations are finished.
        A call to :meth:`~Surface.mark_dirty` is required
        after the data is modified.

        :returns: A read-write CFFI buffer object.

        """
        return ffi.buffer(cairo.cairo_image_surface_get_data(self._pointer), self.get_stride() * self.get_height())

    def get_format(self):
        """Return the :ref:`FORMAT` string of the surface."""
        return cairo.cairo_image_surface_get_format(self._pointer)

    def get_width(self):
        """Return the width of the surface, in pixels."""
        return cairo.cairo_image_surface_get_width(self._pointer)

    def get_height(self):
        """Return the width of the surface, in pixels."""
        return cairo.cairo_image_surface_get_height(self._pointer)

    def get_stride(self):
        """Return the stride of the image surface in bytes
        (or 0 if surface is not an image surface).

        The stride is the distance in bytes
        from the beginning of one row of the image data
        to the beginning of the next row.

        """
        return cairo.cairo_image_surface_get_stride(self._pointer)