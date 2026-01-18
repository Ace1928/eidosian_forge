import os
import sys
import ctypes
import threading
import logging
import numpy
from ..core import (
def _wrap_bitmap_bits_in_array(self, shape, dtype, save):
    """Return an ndarray view on the data in a FreeImage bitmap. Only
        valid for as long as the bitmap is loaded (if single page) / locked
        in memory (if multipage). This is used in loading data, but
        also during saving, to prepare a strided numpy array buffer.

        """
    with self._fi as lib:
        pitch = lib.FreeImage_GetPitch(self._bitmap)
        bits = lib.FreeImage_GetBits(self._bitmap)
    height = shape[-1]
    byte_size = height * pitch
    itemsize = dtype.itemsize
    if len(shape) == 3:
        strides = (itemsize, shape[0] * itemsize, pitch)
    else:
        strides = (itemsize, pitch)
    data = (ctypes.c_char * byte_size).from_address(bits)
    try:
        self._need_finish = False
        if TEST_NUMPY_NO_STRIDES:
            raise NotImplementedError()
        return numpy.ndarray(shape, dtype=dtype, buffer=data, strides=strides)
    except NotImplementedError:
        if save:
            self._need_finish = True
            return numpy.zeros(shape, dtype=dtype)
        else:
            bb = bytes(bytearray(data))
            array = numpy.frombuffer(bb, dtype=dtype).copy()
            if len(shape) == 3:
                array.shape = (shape[2], strides[-1] // shape[0], shape[0])
                array2 = array[:shape[2], :shape[1], :shape[0]]
                array = numpy.zeros(shape, dtype=array.dtype)
                for i in range(shape[0]):
                    array[i] = array2[:, :, i].T
            else:
                array.shape = (shape[1], strides[-1])
                array = array[:shape[1], :shape[0]].T
            return array