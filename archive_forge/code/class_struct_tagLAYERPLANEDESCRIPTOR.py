from ctypes import *
from ctypes import _SimpleCData, _check_size
from OpenGL import extensions
from OpenGL.raw.GL._types import *
from OpenGL._bytes import as_8_bit
from OpenGL._opaque import opaque_pointer_cls as _opaque_pointer_cls
class struct_tagLAYERPLANEDESCRIPTOR(Structure):
    __slots__ = ['nSize', 'nVersion', 'dwFlags', 'iPixelType', 'cColorBits', 'cRedBits', 'cRedShift', 'cGreenBits', 'cGreenShift', 'cBlueBits', 'cBlueShift', 'cAlphaBits', 'cAlphaShift', 'cAccumBits', 'cAccumRedBits', 'cAccumGreenBits', 'cAccumBlueBits', 'cAccumAlphaBits', 'cDepthBits', 'cStencilBits', 'cAuxBuffers', 'iLayerPlane', 'bReserved', 'crTransparent']