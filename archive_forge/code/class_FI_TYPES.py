import os
import sys
import ctypes
import threading
import logging
import numpy
from ..core import (
class FI_TYPES(object):
    FIT_UNKNOWN = 0
    FIT_BITMAP = 1
    FIT_UINT16 = 2
    FIT_INT16 = 3
    FIT_UINT32 = 4
    FIT_INT32 = 5
    FIT_FLOAT = 6
    FIT_DOUBLE = 7
    FIT_COMPLEX = 8
    FIT_RGB16 = 9
    FIT_RGBA16 = 10
    FIT_RGBF = 11
    FIT_RGBAF = 12
    dtypes = {FIT_BITMAP: numpy.uint8, FIT_UINT16: numpy.uint16, FIT_INT16: numpy.int16, FIT_UINT32: numpy.uint32, FIT_INT32: numpy.int32, FIT_FLOAT: numpy.float32, FIT_DOUBLE: numpy.float64, FIT_COMPLEX: numpy.complex128, FIT_RGB16: numpy.uint16, FIT_RGBA16: numpy.uint16, FIT_RGBF: numpy.float32, FIT_RGBAF: numpy.float32}
    fi_types = {(numpy.uint8, 1): FIT_BITMAP, (numpy.uint8, 3): FIT_BITMAP, (numpy.uint8, 4): FIT_BITMAP, (numpy.uint16, 1): FIT_UINT16, (numpy.int16, 1): FIT_INT16, (numpy.uint32, 1): FIT_UINT32, (numpy.int32, 1): FIT_INT32, (numpy.float32, 1): FIT_FLOAT, (numpy.float64, 1): FIT_DOUBLE, (numpy.complex128, 1): FIT_COMPLEX, (numpy.uint16, 3): FIT_RGB16, (numpy.uint16, 4): FIT_RGBA16, (numpy.float32, 3): FIT_RGBF, (numpy.float32, 4): FIT_RGBAF}
    extra_dims = {FIT_UINT16: [], FIT_INT16: [], FIT_UINT32: [], FIT_INT32: [], FIT_FLOAT: [], FIT_DOUBLE: [], FIT_COMPLEX: [], FIT_RGB16: [3], FIT_RGBA16: [4], FIT_RGBF: [3], FIT_RGBAF: [4]}