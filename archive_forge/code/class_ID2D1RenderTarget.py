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
class ID2D1RenderTarget(ID2D1Resource, com.pIUnknown):
    _methods_ = [('CreateBitmap', com.STDMETHOD()), ('CreateBitmapFromWicBitmap', com.STDMETHOD()), ('CreateSharedBitmap', com.STDMETHOD()), ('CreateBitmapBrush', com.STDMETHOD()), ('CreateSolidColorBrush', com.STDMETHOD(POINTER(D2D1_COLOR_F), c_void_p, POINTER(ID2D1SolidColorBrush))), ('CreateGradientStopCollection', com.STDMETHOD()), ('CreateLinearGradientBrush', com.STDMETHOD()), ('CreateRadialGradientBrush', com.STDMETHOD()), ('CreateCompatibleRenderTarget', com.STDMETHOD()), ('CreateLayer', com.STDMETHOD()), ('CreateMesh', com.STDMETHOD()), ('DrawLine', com.STDMETHOD()), ('DrawRectangle', com.STDMETHOD()), ('FillRectangle', com.STDMETHOD()), ('DrawRoundedRectangle', com.STDMETHOD()), ('FillRoundedRectangle', com.STDMETHOD()), ('DrawEllipse', com.STDMETHOD()), ('FillEllipse', com.STDMETHOD()), ('DrawGeometry', com.STDMETHOD()), ('FillGeometry', com.STDMETHOD()), ('FillMesh', com.STDMETHOD()), ('FillOpacityMask', com.STDMETHOD()), ('DrawBitmap', com.STDMETHOD()), ('DrawText', com.STDMETHOD(c_wchar_p, UINT, IDWriteTextFormat, POINTER(D2D1_RECT_F), ID2D1Brush, D2D1_DRAW_TEXT_OPTIONS, DWRITE_MEASURING_MODE)), ('DrawTextLayout', com.METHOD(c_void, D2D_POINT_2F, IDWriteTextLayout, ID2D1Brush, UINT32)), ('DrawGlyphRun', com.METHOD(c_void, D2D_POINT_2F, POINTER(DWRITE_GLYPH_RUN), ID2D1Brush, UINT32)), ('SetTransform', com.METHOD(c_void)), ('GetTransform', com.STDMETHOD()), ('SetAntialiasMode', com.METHOD(c_void, D2D1_TEXT_ANTIALIAS_MODE)), ('GetAntialiasMode', com.STDMETHOD()), ('SetTextAntialiasMode', com.METHOD(c_void, D2D1_TEXT_ANTIALIAS_MODE)), ('GetTextAntialiasMode', com.STDMETHOD()), ('SetTextRenderingParams', com.STDMETHOD(IDWriteRenderingParams)), ('GetTextRenderingParams', com.STDMETHOD()), ('SetTags', com.STDMETHOD()), ('GetTags', com.STDMETHOD()), ('PushLayer', com.STDMETHOD()), ('PopLayer', com.STDMETHOD()), ('Flush', com.STDMETHOD(c_void_p, c_void_p)), ('SaveDrawingState', com.STDMETHOD()), ('RestoreDrawingState', com.STDMETHOD()), ('PushAxisAlignedClip', com.STDMETHOD()), ('PopAxisAlignedClip', com.STDMETHOD()), ('Clear', com.METHOD(c_void, POINTER(D2D1_COLOR_F))), ('BeginDraw', com.METHOD(c_void)), ('EndDraw', com.STDMETHOD(c_void_p, c_void_p)), ('GetPixelFormat', com.STDMETHOD()), ('SetDpi', com.STDMETHOD()), ('GetDpi', com.STDMETHOD()), ('GetSize', com.STDMETHOD()), ('GetPixelSize', com.STDMETHOD()), ('GetMaximumBitmapSize', com.STDMETHOD()), ('IsSupported', com.STDMETHOD())]