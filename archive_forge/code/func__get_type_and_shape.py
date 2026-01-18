import os
import sys
import ctypes
import threading
import logging
import numpy
from ..core import (
def _get_type_and_shape(self):
    bitmap = self._bitmap
    with self._fi as lib:
        w = lib.FreeImage_GetWidth(bitmap)
        h = lib.FreeImage_GetHeight(bitmap)
        self._fi_type = fi_type = lib.FreeImage_GetImageType(bitmap)
        if not fi_type:
            raise ValueError('Unknown image pixel type')
    bpp = None
    dtype = FI_TYPES.dtypes[fi_type]
    if fi_type == FI_TYPES.FIT_BITMAP:
        with self._fi as lib:
            bpp = lib.FreeImage_GetBPP(bitmap)
            has_pallette = lib.FreeImage_GetColorsUsed(bitmap)
        if has_pallette:
            if has_pallette == 256:
                palette = lib.FreeImage_GetPalette(bitmap)
                palette = ctypes.c_void_p(palette)
                p = (ctypes.c_uint8 * (256 * 4)).from_address(palette.value)
                p = numpy.frombuffer(p, numpy.uint32).copy()
                if (GREY_PALETTE == p).all():
                    extra_dims = []
                    return (numpy.dtype(dtype), extra_dims + [w, h], bpp)
            newbitmap = lib.FreeImage_ConvertTo32Bits(bitmap)
            newbitmap = ctypes.c_void_p(newbitmap)
            self._set_bitmap(newbitmap)
            return self._get_type_and_shape()
        elif bpp == 8:
            extra_dims = []
        elif bpp == 24:
            extra_dims = [3]
        elif bpp == 32:
            extra_dims = [4]
        else:
            newbitmap = lib.FreeImage_ConvertTo32Bits(bitmap)
            newbitmap = ctypes.c_void_p(newbitmap)
            self._set_bitmap(newbitmap)
            return self._get_type_and_shape()
    else:
        extra_dims = FI_TYPES.extra_dims[fi_type]
    return (numpy.dtype(dtype), extra_dims + [w, h], bpp)