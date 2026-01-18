import os
import sys
import ctypes
import threading
import logging
import numpy
from ..core import (
class METADATA_DATATYPE(object):
    FIDT_BYTE = 1
    FIDT_ASCII = 2
    FIDT_SHORT = 3
    FIDT_LONG = 4
    FIDT_RATIONAL = 5
    FIDT_SBYTE = 6
    FIDT_UNDEFINED = 7
    FIDT_SSHORT = 8
    FIDT_SLONG = 9
    FIDT_SRATIONAL = 10
    FIDT_FLOAT = 11
    FIDT_DOUBLE = 12
    FIDT_IFD = 13
    FIDT_PALETTE = 14
    FIDT_LONG8 = 16
    FIDT_SLONG8 = 17
    FIDT_IFD8 = 18
    dtypes = {FIDT_BYTE: numpy.uint8, FIDT_SHORT: numpy.uint16, FIDT_LONG: numpy.uint32, FIDT_RATIONAL: [('numerator', numpy.uint32), ('denominator', numpy.uint32)], FIDT_LONG8: numpy.uint64, FIDT_SLONG8: numpy.int64, FIDT_IFD8: numpy.uint64, FIDT_SBYTE: numpy.int8, FIDT_UNDEFINED: numpy.uint8, FIDT_SSHORT: numpy.int16, FIDT_SLONG: numpy.int32, FIDT_SRATIONAL: [('numerator', numpy.int32), ('denominator', numpy.int32)], FIDT_FLOAT: numpy.float32, FIDT_DOUBLE: numpy.float64, FIDT_IFD: numpy.uint32, FIDT_PALETTE: [('R', numpy.uint8), ('G', numpy.uint8), ('B', numpy.uint8), ('A', numpy.uint8)]}