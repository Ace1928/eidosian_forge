import array
import base64
import contextlib
import gc
import io
import math
import os
import shutil
import sys
import tempfile
import cairocffi
import pikepdf
import pytest
from . import (
@pytest.mark.xfail(cairo_version() < 11000, reason='Cairo version too low')
def _recording_surface_common(extents):
    surface = ImageSurface(cairocffi.FORMAT_ARGB32, 100, 100)
    empty_pixels = surface.get_data()[:]
    assert empty_pixels == b'\x00' * 40000
    surface = ImageSurface(cairocffi.FORMAT_ARGB32, 100, 100)
    context = Context(surface)
    context.move_to(20, 50)
    context.show_text('Something about us.')
    text_pixels = surface.get_data()[:]
    assert text_pixels != empty_pixels
    recording_surface = RecordingSurface(cairocffi.CONTENT_COLOR_ALPHA, extents)
    context = Context(recording_surface)
    context.move_to(20, 50)
    assert recording_surface.ink_extents() == (0, 0, 0, 0)
    context.show_text('Something about us.')
    recording_surface.flush()
    assert recording_surface.ink_extents() != (0, 0, 0, 0)
    surface = ImageSurface(cairocffi.FORMAT_ARGB32, 100, 100)
    context = Context(surface)
    context.set_source_surface(recording_surface)
    context.paint()
    recorded_pixels = surface.get_data()[:]
    return (text_pixels, recorded_pixels)