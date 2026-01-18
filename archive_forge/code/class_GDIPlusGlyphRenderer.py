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
class GDIPlusGlyphRenderer(base.GlyphRenderer):

    def __init__(self, font: 'GDIPlusFont') -> None:
        self._bitmap = None
        self._dc = None
        self._bitmap_rect = None
        super().__init__(font)
        self.font = font
        width = font.max_glyph_width
        height = font.ascent - font.descent
        width = (width | 3) + 1
        height = (height | 3) + 1
        self._create_bitmap(width, height)
        gdi32.SelectObject(self._dc, self.font.hfont)

    def __del__(self) -> None:
        try:
            if self._matrix:
                gdiplus.GdipDeleteMatrix(self._matrix)
            if self._brush:
                gdiplus.GdipDeleteBrush(self._brush)
            if self._graphics:
                gdiplus.GdipDeleteGraphics(self._graphics)
            if self._bitmap:
                gdiplus.GdipDisposeImage(self._bitmap)
            if self._dc:
                user32.ReleaseDC(0, self._dc)
        except Exception:
            pass

    def _create_bitmap(self, width: int, height: int) -> None:
        self._data = (BYTE * (4 * width * height))()
        self._bitmap = ctypes.c_void_p()
        self._format = PixelFormat32bppARGB
        gdiplus.GdipCreateBitmapFromScan0(width, height, width * 4, self._format, self._data, ctypes.byref(self._bitmap))
        self._graphics = ctypes.c_void_p()
        gdiplus.GdipGetImageGraphicsContext(self._bitmap, ctypes.byref(self._graphics))
        gdiplus.GdipSetPageUnit(self._graphics, UnitPixel)
        self._dc = user32.GetDC(0)
        gdi32.SelectObject(self._dc, self.font.hfont)
        gdiplus.GdipSetTextRenderingHint(self._graphics, TextRenderingHintAntiAliasGridFit)
        self._brush = ctypes.c_void_p()
        gdiplus.GdipCreateSolidFill(4294967295, ctypes.byref(self._brush))
        self._matrix = ctypes.c_void_p()
        gdiplus.GdipCreateMatrix(ctypes.byref(self._matrix))
        self._flags = DriverStringOptionsCmapLookup | DriverStringOptionsRealizedAdvance
        self._rect = Rect(0, 0, width, height)
        self._bitmap_height = height

    def render(self, text: str) -> Glyph:
        ch = ctypes.create_unicode_buffer(text)
        len_ch = len(text)
        width = 10000
        height = self._bitmap_height
        rect = Rectf(0, self._bitmap_height - self.font.ascent + self.font.descent, width, height)
        generic = ctypes.c_void_p()
        gdiplus.GdipStringFormatGetGenericTypographic(ctypes.byref(generic))
        fmt = ctypes.c_void_p()
        gdiplus.GdipCloneStringFormat(generic, ctypes.byref(fmt))
        gdiplus.GdipDeleteStringFormat(generic)
        bbox = Rectf()
        flags = StringFormatFlagsMeasureTrailingSpaces | StringFormatFlagsNoClip | StringFormatFlagsNoFitBlackBox
        gdiplus.GdipSetStringFormatFlags(fmt, flags)
        gdiplus.GdipMeasureString(self._graphics, ch, len_ch, self.font._gdipfont, ctypes.byref(rect), fmt, ctypes.byref(bbox), None, None)
        advance = int(math.ceil(bbox.width))
        if text == '\r\n':
            text = '\r'
        abc = ABC()
        width = 0
        lsb = 0
        ttf_font = True
        for codepoint in [ord(c) for c in text]:
            if gdi32.GetCharABCWidthsW(self._dc, codepoint, codepoint, ctypes.byref(abc)):
                lsb += abc.abcA
                width += abc.abcB
                if lsb < 0:
                    rect.x = -lsb
                    width -= lsb
                else:
                    width += lsb
            else:
                ttf_font = False
                break
        if not ttf_font:
            width = advance
            if self.font.italic:
                width += width // 2
                width = min(width, self._rect.Width)
        gdiplus.GdipGraphicsClear(self._graphics, 0)
        gdiplus.GdipDrawString(self._graphics, ch, len_ch, self.font._gdipfont, ctypes.byref(rect), fmt, self._brush)
        gdiplus.GdipFlush(self._graphics, 1)
        gdiplus.GdipDeleteStringFormat(fmt)
        bitmap_data = BitmapData()
        gdiplus.GdipBitmapLockBits(self._bitmap, ctypes.byref(self._rect), ImageLockModeRead, self._format, ctypes.byref(bitmap_data))
        buffer = ctypes.create_string_buffer(bitmap_data.Stride * bitmap_data.Height)
        ctypes.memmove(buffer, bitmap_data.Scan0, len(buffer))
        gdiplus.GdipBitmapUnlockBits(self._bitmap, ctypes.byref(bitmap_data))
        image = pyglet.image.ImageData(width, height, 'BGRA', buffer, -bitmap_data.Stride)
        glyph = self.font.create_glyph(image)
        lsb = min(lsb, 0)
        glyph.set_bearings(-self.font.descent, lsb, advance)
        return glyph