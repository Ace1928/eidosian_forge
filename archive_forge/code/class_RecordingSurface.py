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
class RecordingSurface(Surface):
    """A recording surface is a surface that records all drawing operations
    at the highest level of the surface backend interface,
    (that is, the level of paint, mask, stroke, fill, and show_text_glyphs).
    The recording surface can then be "replayed" against any target surface
    by using it as a source surface.

    If you want to replay a surface so that the results in ``target``
    will be identical to the results that would have been obtained
    if the original operations applied to the recording surface
    had instead been applied to the target surface,
    you can use code like this::

        context = Context(target)
        context.set_source_surface(recording_surface, 0, 0)
        context.paint()

    A recording surface is logically unbounded,
    i.e. it has no implicit constraint on the size of the drawing surface.
    However, in practice this is rarely useful as you wish to replay
    against a particular target surface with known bounds.
    For this case, it is more efficient to specify the target extents
    to the recording surface upon creation.

    The recording phase of the recording surface is careful
    to snapshot all necessary objects (paths, patterns, etc.),
    in order to achieve accurate replay.

    :param content: The :ref:`CONTENT` string of the recording surface
    :param extents:
        The extents to record
        as a ``(x, y, width, height)`` tuple of floats in device units,
        or :obj:`None` to record unbounded operations.
        ``(x, y)`` are the coordinates of the top-left corner of the rectangle,
        ``(width, height)`` its dimensions.

    *New in cairo 1.10*

    *New in cairocffi 0.2*

    """

    def __init__(self, content, extents):
        extents = ffi.new('cairo_rectangle_t *', extents) if extents is not None else ffi.NULL
        Surface.__init__(self, cairo.cairo_recording_surface_create(content, extents))

    def get_extents(self):
        """Return the extents of the recording-surface.

        :returns:
            A ``(x, y, width, height)`` tuple of floats,
            or :obj:`None` if the surface is unbounded.

        *New in cairo 1.12*

        """
        extents = ffi.new('cairo_rectangle_t *')
        if cairo.cairo_recording_surface_get_extents(self._pointer, extents):
            return (extents.x, extents.y, extents.width, extents.height)

    def ink_extents(self):
        """Measures the extents of the operations
        stored within the recording-surface.
        This is useful to compute the required size of an image surface
        (or equivalent) into which to replay the full sequence
        of drawing operations.

        :return: A ``(x, y, width, height)`` tuple of floats.

        """
        extents = ffi.new('double[4]')
        cairo.cairo_recording_surface_ink_extents(self._pointer, extents + 0, extents + 1, extents + 2, extents + 3)
        self._check_status()
        return tuple(extents)