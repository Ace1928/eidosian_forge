from ctypes import *
from functools import lru_cache
import unicodedata
from pyglet import compat_platform
import pyglet
from pyglet.window import BaseWindow, WindowException, MouseCursor
from pyglet.window import DefaultMouseCursor, _PlatformEventHandler, _ViewEventHandler
from pyglet.event import EventDispatcher
from pyglet.window import key, mouse
from pyglet.canvas.win32 import Win32Canvas
from pyglet.libs.win32 import _user32, _kernel32, _gdi32, _dwmapi, _shell32
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.winkey import *
from pyglet.libs.win32.types import *
@lru_cache()
def _create_cursor_from_image(self, cursor):
    """Creates platform cursor from an ImageCursor instance."""
    fmt = 'BGRA'
    image = cursor.texture
    pitch = len(fmt) * image.width
    header = BITMAPINFOHEADER()
    header.biSize = sizeof(header)
    header.biWidth = image.width
    header.biHeight = image.height
    header.biPlanes = 1
    header.biBitCount = 32
    hdc = _user32.GetDC(None)
    dataptr = c_void_p()
    bitmap = _gdi32.CreateDIBSection(hdc, byref(header), DIB_RGB_COLORS, byref(dataptr), None, 0)
    _user32.ReleaseDC(None, hdc)
    image = image.get_image_data()
    data = image.get_data(fmt, pitch)
    memmove(dataptr, data, len(data))
    mask = _gdi32.CreateBitmap(image.width, image.height, 1, 1, None)
    iconinfo = ICONINFO()
    iconinfo.fIcon = False
    iconinfo.hbmMask = mask
    iconinfo.hbmColor = bitmap
    iconinfo.xHotspot = int(cursor.hot_x)
    iconinfo.yHotspot = int(image.height - cursor.hot_y)
    icon = _user32.CreateIconIndirect(byref(iconinfo))
    _gdi32.DeleteObject(mask)
    _gdi32.DeleteObject(bitmap)
    return icon