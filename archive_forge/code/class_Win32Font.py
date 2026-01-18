from __future__ import annotations
import ctypes
import math
import warnings
from typing import Optional, Sequence, TYPE_CHECKING
import pyglet
import pyglet.image
from pyglet.font import base
from pyglet.image.codecs.gdiplus import ImageLockModeRead, BitmapData
from pyglet.image.codecs.gdiplus import PixelFormat32bppARGB, gdiplus, Rect
from pyglet.libs.win32 import _gdi32 as gdi32, _user32 as user32
from pyglet.libs.win32.types import BYTE, ABC, TEXTMETRIC, LOGFONTW
from pyglet.libs.win32.constants import FW_BOLD, FW_NORMAL, ANTIALIASED_QUALITY
from pyglet.libs.win32.context_managers import device_context
class Win32Font(base.Font):
    glyph_renderer_class = GDIPlusGlyphRenderer

    def __init__(self, name: str, size: float, bold: bool=False, italic: bool=False, stretch: bool=False, dpi: Optional[float]=None) -> None:
        super().__init__()
        self.logfont = self.get_logfont(name, size, bold, italic, dpi)
        self.hfont = gdi32.CreateFontIndirectW(ctypes.byref(self.logfont))
        with device_context(None) as dc:
            metrics = TEXTMETRIC()
            gdi32.SelectObject(dc, self.hfont)
            gdi32.GetTextMetricsA(dc, ctypes.byref(metrics))
            self.ascent = metrics.tmAscent
            self.descent = -metrics.tmDescent
            self.max_glyph_width = metrics.tmMaxCharWidth

    @staticmethod
    def get_logfont(name: str, size: float, bold: bool, italic: bool, dpi: Optional[float]=None) -> LOGFONTW:
        """Get a raw Win32 :py:class:`.LOGFONTW` struct for the given arguments.

        Args:
            name: The name of the font
            size: The font size in points
            bold: Whether to request the font as bold
            italic: Whether to request the font as italic
            dpi: The screen dpi

        Returns:
            LOGFONTW: a ctypes binding of a Win32 LOGFONTW struct
        """
        with device_context(None) as dc:
            if dpi is None:
                dpi = 96
            log_pixels_y = dpi
            logfont = LOGFONTW()
            logfont.lfHeight = int(-size * log_pixels_y // 72)
            if bold:
                logfont.lfWeight = FW_BOLD
            else:
                logfont.lfWeight = FW_NORMAL
            logfont.lfItalic = italic
            logfont.lfFaceName = name
            logfont.lfQuality = ANTIALIASED_QUALITY
        return logfont