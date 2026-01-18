import os
import unittest
from pygame.tests import test_utils
from pygame.tests.test_utils import (
import pygame
from pygame.locals import *
from pygame.bufferproxy import BufferProxy
import platform
import gc
import weakref
import ctypes
def _check_interface_3D(self, s):
    s_w, s_h = s.get_size()
    s_bytesize = s.get_bytesize()
    s_pitch = s.get_pitch()
    s_pixels = s._pixels_address
    s_shifts = list(s.get_shifts())
    if s_shifts[0:3] == [0, 8, 16]:
        if self.lilendian:
            offset = 0
            step = 1
        else:
            offset = s_bytesize - 1
            step = -1
    elif s_shifts[0:3] == [8, 16, 24]:
        if self.lilendian:
            offset = 1
            step = 1
        else:
            offset = s_bytesize - 2
            step = -1
    elif s_shifts[0:3] == [16, 8, 0]:
        if self.lilendian:
            offset = 2
            step = -1
        else:
            offset = s_bytesize - 3
            step = 1
    elif s_shifts[0:3] == [24, 16, 8]:
        if self.lilendian:
            offset = 2
            step = -1
        else:
            offset = s_bytesize - 4
            step = -1
    else:
        return
    v = s.get_view('3')
    if not IS_PYPY:
        inter = ArrayInterface(v)
        flags = PAI_ALIGNED | PAI_NOTSWAPPED | PAI_WRITEABLE
        self.assertEqual(inter.two, 2)
        self.assertEqual(inter.nd, 3)
        self.assertEqual(inter.typekind, 'u')
        self.assertEqual(inter.itemsize, 1)
        self.assertEqual(inter.shape[0], s_w)
        self.assertEqual(inter.shape[1], s_h)
        self.assertEqual(inter.shape[2], 3)
        self.assertEqual(inter.strides[0], s_bytesize)
        self.assertEqual(inter.strides[1], s_pitch)
        self.assertEqual(inter.strides[2], step)
        self.assertEqual(inter.flags, flags)
        self.assertEqual(inter.data, s_pixels + offset)