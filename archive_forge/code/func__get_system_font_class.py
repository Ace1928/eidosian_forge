import os
import sys
import weakref
from typing import Union, BinaryIO, Optional, Iterable
import pyglet
from pyglet.font.user import UserDefinedFontBase
from pyglet import gl
def _get_system_font_class():
    """Get the appropriate class for the system being used.

    Pyglet relies on OS dependent font systems for loading fonts and glyph creation.
    """
    if pyglet.compat_platform == 'darwin':
        from pyglet.font.quartz import QuartzFont
        _font_class = QuartzFont
    elif pyglet.compat_platform in ('win32', 'cygwin'):
        from pyglet.libs.win32.constants import WINDOWS_7_OR_GREATER
        if WINDOWS_7_OR_GREATER and (not pyglet.options['win32_gdi_font']):
            from pyglet.font.directwrite import Win32DirectWriteFont
            _font_class = Win32DirectWriteFont
        else:
            from pyglet.font.win32 import GDIPlusFont
            _font_class = GDIPlusFont
    else:
        from pyglet.font.freetype import FreeTypeFont
        _font_class = FreeTypeFont
    return _font_class