import copy
import os
import pathlib
import platform
from ctypes import *
from typing import List, Optional, Tuple
import math
import pyglet
from pyglet.font import base
from pyglet.image.codecs.wic import IWICBitmap, WICDecoder, GUID_WICPixelFormat32bppPBGRA
from pyglet.libs.win32 import _kernel32 as kernel32
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
from pyglet.util import debug_print
class ID2D1Factory(com.pIUnknown):
    _methods_ = [('ReloadSystemMetrics', com.STDMETHOD()), ('GetDesktopDpi', com.STDMETHOD()), ('CreateRectangleGeometry', com.STDMETHOD()), ('CreateRoundedRectangleGeometry', com.STDMETHOD()), ('CreateEllipseGeometry', com.STDMETHOD()), ('CreateGeometryGroup', com.STDMETHOD()), ('CreateTransformedGeometry', com.STDMETHOD()), ('CreatePathGeometry', com.STDMETHOD()), ('CreateStrokeStyle', com.STDMETHOD()), ('CreateDrawingStateBlock', com.STDMETHOD()), ('CreateWicBitmapRenderTarget', com.STDMETHOD(IWICBitmap, POINTER(D2D1_RENDER_TARGET_PROPERTIES), POINTER(ID2D1RenderTarget))), ('CreateHwndRenderTarget', com.STDMETHOD()), ('CreateDxgiSurfaceRenderTarget', com.STDMETHOD()), ('CreateDCRenderTarget', com.STDMETHOD())]