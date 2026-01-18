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
class SurfaceGetBufferTest(unittest.TestCase):
    try:
        ArrayInterface
    except NameError:
        __tags__ = ('ignore', 'subprocess_ignore')
    lilendian = pygame.get_sdl_byteorder() == pygame.LIL_ENDIAN

    def _check_interface_2D(self, s):
        s_w, s_h = s.get_size()
        s_bytesize = s.get_bytesize()
        s_pitch = s.get_pitch()
        s_pixels = s._pixels_address
        v = s.get_view('2')
        if not IS_PYPY:
            flags = PAI_ALIGNED | PAI_NOTSWAPPED | PAI_WRITEABLE
            if s.get_pitch() == s_w * s_bytesize:
                flags |= PAI_FORTRAN
            inter = ArrayInterface(v)
            self.assertEqual(inter.two, 2)
            self.assertEqual(inter.nd, 2)
            self.assertEqual(inter.typekind, 'u')
            self.assertEqual(inter.itemsize, s_bytesize)
            self.assertEqual(inter.shape[0], s_w)
            self.assertEqual(inter.shape[1], s_h)
            self.assertEqual(inter.strides[0], s_bytesize)
            self.assertEqual(inter.strides[1], s_pitch)
            self.assertEqual(inter.flags, flags)
            self.assertEqual(inter.data, s_pixels)

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

    def _check_interface_rgba(self, s, plane):
        s_w, s_h = s.get_size()
        s_bytesize = s.get_bytesize()
        s_pitch = s.get_pitch()
        s_pixels = s._pixels_address
        s_shifts = s.get_shifts()
        s_masks = s.get_masks()
        if not s_masks[plane]:
            return
        alpha_shift = s_shifts[plane]
        offset = alpha_shift // 8
        if not self.lilendian:
            offset = s_bytesize - offset - 1
        v = s.get_view('rgba'[plane])
        if not IS_PYPY:
            inter = ArrayInterface(v)
            flags = PAI_ALIGNED | PAI_NOTSWAPPED | PAI_WRITEABLE
            self.assertEqual(inter.two, 2)
            self.assertEqual(inter.nd, 2)
            self.assertEqual(inter.typekind, 'u')
            self.assertEqual(inter.itemsize, 1)
            self.assertEqual(inter.shape[0], s_w)
            self.assertEqual(inter.shape[1], s_h)
            self.assertEqual(inter.strides[0], s_bytesize)
            self.assertEqual(inter.strides[1], s_pitch)
            self.assertEqual(inter.flags, flags)
            self.assertEqual(inter.data, s_pixels + offset)

    def test_array_interface(self):
        self._check_interface_2D(pygame.Surface((5, 7), 0, 8))
        self._check_interface_2D(pygame.Surface((5, 7), 0, 16))
        self._check_interface_2D(pygame.Surface((5, 7), pygame.SRCALPHA, 16))
        self._check_interface_3D(pygame.Surface((5, 7), 0, 24))
        self._check_interface_3D(pygame.Surface((8, 4), 0, 24))
        self._check_interface_2D(pygame.Surface((5, 7), 0, 32))
        self._check_interface_3D(pygame.Surface((5, 7), 0, 32))
        self._check_interface_2D(pygame.Surface((5, 7), pygame.SRCALPHA, 32))
        self._check_interface_3D(pygame.Surface((5, 7), pygame.SRCALPHA, 32))

    def test_array_interface_masks(self):
        """Test non-default color byte orders on 3D views"""
        sz = (5, 7)
        s = pygame.Surface(sz, 0, 32)
        s_masks = list(s.get_masks())
        masks = [255, 65280, 16711680]
        if s_masks[0:3] == masks or s_masks[0:3] == masks[::-1]:
            masks = s_masks[2::-1] + s_masks[3:4]
            self._check_interface_3D(pygame.Surface(sz, 0, 32, masks))
        s = pygame.Surface(sz, 0, 24)
        s_masks = list(s.get_masks())
        masks = [255, 65280, 16711680]
        if s_masks[0:3] == masks or s_masks[0:3] == masks[::-1]:
            masks = s_masks[2::-1] + s_masks[3:4]
            self._check_interface_3D(pygame.Surface(sz, 0, 24, masks))
        masks = [65280, 16711680, 4278190080, 0]
        self._check_interface_3D(pygame.Surface(sz, 0, 32, masks))

    def test_array_interface_alpha(self):
        for shifts in [[0, 8, 16, 24], [8, 16, 24, 0], [24, 16, 8, 0], [16, 8, 0, 24]]:
            masks = [255 << s for s in shifts]
            s = pygame.Surface((4, 2), pygame.SRCALPHA, 32, masks)
            self._check_interface_rgba(s, 3)

    def test_array_interface_rgb(self):
        for shifts in [[0, 8, 16, 24], [8, 16, 24, 0], [24, 16, 8, 0], [16, 8, 0, 24]]:
            masks = [255 << s for s in shifts]
            masks[3] = 0
            for plane in range(3):
                s = pygame.Surface((4, 2), 0, 24)
                self._check_interface_rgba(s, plane)
                s = pygame.Surface((4, 2), 0, 32)
                self._check_interface_rgba(s, plane)

    @unittest.skipIf(not pygame.HAVE_NEWBUF, 'newbuf not implemented')
    def test_newbuf_PyBUF_flags_bytes(self):
        from pygame.tests.test_utils import buftools
        Importer = buftools.Importer
        s = pygame.Surface((10, 6), 0, 32)
        a = s.get_buffer()
        b = Importer(a, buftools.PyBUF_SIMPLE)
        self.assertEqual(b.ndim, 0)
        self.assertTrue(b.format is None)
        self.assertEqual(b.len, a.length)
        self.assertEqual(b.itemsize, 1)
        self.assertTrue(b.shape is None)
        self.assertTrue(b.strides is None)
        self.assertTrue(b.suboffsets is None)
        self.assertFalse(b.readonly)
        self.assertEqual(b.buf, s._pixels_address)
        b = Importer(a, buftools.PyBUF_WRITABLE)
        self.assertEqual(b.ndim, 0)
        self.assertTrue(b.format is None)
        self.assertFalse(b.readonly)
        b = Importer(a, buftools.PyBUF_FORMAT)
        self.assertEqual(b.ndim, 0)
        self.assertEqual(b.format, 'B')
        b = Importer(a, buftools.PyBUF_ND)
        self.assertEqual(b.ndim, 1)
        self.assertTrue(b.format is None)
        self.assertEqual(b.len, a.length)
        self.assertEqual(b.itemsize, 1)
        self.assertEqual(b.shape, (a.length,))
        self.assertTrue(b.strides is None)
        self.assertTrue(b.suboffsets is None)
        self.assertFalse(b.readonly)
        self.assertEqual(b.buf, s._pixels_address)
        b = Importer(a, buftools.PyBUF_STRIDES)
        self.assertEqual(b.ndim, 1)
        self.assertTrue(b.format is None)
        self.assertEqual(b.strides, (1,))
        s2 = s.subsurface((1, 1, 7, 4))
        a = s2.get_buffer()
        b = Importer(a, buftools.PyBUF_SIMPLE)
        self.assertEqual(b.ndim, 0)
        self.assertTrue(b.format is None)
        self.assertEqual(b.len, a.length)
        self.assertEqual(b.itemsize, 1)
        self.assertTrue(b.shape is None)
        self.assertTrue(b.strides is None)
        self.assertTrue(b.suboffsets is None)
        self.assertFalse(b.readonly)
        self.assertEqual(b.buf, s2._pixels_address)
        b = Importer(a, buftools.PyBUF_C_CONTIGUOUS)
        self.assertEqual(b.ndim, 1)
        self.assertEqual(b.strides, (1,))
        b = Importer(a, buftools.PyBUF_F_CONTIGUOUS)
        self.assertEqual(b.ndim, 1)
        self.assertEqual(b.strides, (1,))
        b = Importer(a, buftools.PyBUF_ANY_CONTIGUOUS)
        self.assertEqual(b.ndim, 1)
        self.assertEqual(b.strides, (1,))

    @unittest.skipIf(not pygame.HAVE_NEWBUF, 'newbuf not implemented')
    def test_newbuf_PyBUF_flags_0D(self):
        from pygame.tests.test_utils import buftools
        Importer = buftools.Importer
        s = pygame.Surface((10, 6), 0, 32)
        a = s.get_view('0')
        b = Importer(a, buftools.PyBUF_SIMPLE)
        self.assertEqual(b.ndim, 0)
        self.assertTrue(b.format is None)
        self.assertEqual(b.len, a.length)
        self.assertEqual(b.itemsize, 1)
        self.assertTrue(b.shape is None)
        self.assertTrue(b.strides is None)
        self.assertTrue(b.suboffsets is None)
        self.assertFalse(b.readonly)
        self.assertEqual(b.buf, s._pixels_address)

    @unittest.skipIf(not pygame.HAVE_NEWBUF, 'newbuf not implemented')
    def test_newbuf_PyBUF_flags_1D(self):
        from pygame.tests.test_utils import buftools
        Importer = buftools.Importer
        s = pygame.Surface((10, 6), 0, 32)
        a = s.get_view('1')
        b = Importer(a, buftools.PyBUF_SIMPLE)
        self.assertEqual(b.ndim, 0)
        self.assertTrue(b.format is None)
        self.assertEqual(b.len, a.length)
        self.assertEqual(b.itemsize, s.get_bytesize())
        self.assertTrue(b.shape is None)
        self.assertTrue(b.strides is None)
        self.assertTrue(b.suboffsets is None)
        self.assertFalse(b.readonly)
        self.assertEqual(b.buf, s._pixels_address)
        b = Importer(a, buftools.PyBUF_WRITABLE)
        self.assertEqual(b.ndim, 0)
        self.assertTrue(b.format is None)
        self.assertFalse(b.readonly)
        b = Importer(a, buftools.PyBUF_FORMAT)
        self.assertEqual(b.ndim, 0)
        self.assertEqual(b.format, '=I')
        b = Importer(a, buftools.PyBUF_ND)
        self.assertEqual(b.ndim, 1)
        self.assertTrue(b.format is None)
        self.assertEqual(b.len, a.length)
        self.assertEqual(b.itemsize, s.get_bytesize())
        self.assertEqual(b.shape, (s.get_width() * s.get_height(),))
        self.assertTrue(b.strides is None)
        self.assertTrue(b.suboffsets is None)
        self.assertFalse(b.readonly)
        self.assertEqual(b.buf, s._pixels_address)
        b = Importer(a, buftools.PyBUF_STRIDES)
        self.assertEqual(b.ndim, 1)
        self.assertTrue(b.format is None)
        self.assertEqual(b.strides, (s.get_bytesize(),))

    @unittest.skipIf(not pygame.HAVE_NEWBUF, 'newbuf not implemented')
    def test_newbuf_PyBUF_flags_2D(self):
        from pygame.tests.test_utils import buftools
        Importer = buftools.Importer
        s = pygame.Surface((10, 6), 0, 32)
        a = s.get_view('2')
        b = Importer(a, buftools.PyBUF_SIMPLE)
        self.assertEqual(b.ndim, 0)
        self.assertTrue(b.format is None)
        self.assertEqual(b.len, a.length)
        self.assertEqual(b.itemsize, s.get_bytesize())
        self.assertTrue(b.shape is None)
        self.assertTrue(b.strides is None)
        self.assertTrue(b.suboffsets is None)
        self.assertFalse(b.readonly)
        self.assertEqual(b.buf, s._pixels_address)
        b = Importer(a, buftools.PyBUF_STRIDES)
        self.assertEqual(b.ndim, 2)
        self.assertTrue(b.format is None)
        self.assertEqual(b.len, a.length)
        self.assertEqual(b.itemsize, s.get_bytesize())
        self.assertEqual(b.shape, s.get_size())
        self.assertEqual(b.strides, (s.get_bytesize(), s.get_pitch()))
        self.assertTrue(b.suboffsets is None)
        self.assertFalse(b.readonly)
        self.assertEqual(b.buf, s._pixels_address)
        b = Importer(a, buftools.PyBUF_RECORDS_RO)
        self.assertEqual(b.ndim, 2)
        self.assertEqual(b.format, '=I')
        self.assertEqual(b.strides, (s.get_bytesize(), s.get_pitch()))
        b = Importer(a, buftools.PyBUF_RECORDS)
        self.assertEqual(b.ndim, 2)
        self.assertEqual(b.format, '=I')
        self.assertEqual(b.strides, (s.get_bytesize(), s.get_pitch()))
        b = Importer(a, buftools.PyBUF_F_CONTIGUOUS)
        self.assertEqual(b.ndim, 2)
        self.assertEqual(b.format, None)
        self.assertEqual(b.strides, (s.get_bytesize(), s.get_pitch()))
        b = Importer(a, buftools.PyBUF_ANY_CONTIGUOUS)
        self.assertEqual(b.ndim, 2)
        self.assertEqual(b.format, None)
        self.assertEqual(b.strides, (s.get_bytesize(), s.get_pitch()))
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_ND)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_C_CONTIGUOUS)
        s2 = s.subsurface((1, 1, 7, 4))
        a = s2.get_view('2')
        b = Importer(a, buftools.PyBUF_STRIDES)
        self.assertEqual(b.ndim, 2)
        self.assertTrue(b.format is None)
        self.assertEqual(b.len, a.length)
        self.assertEqual(b.itemsize, s2.get_bytesize())
        self.assertEqual(b.shape, s2.get_size())
        self.assertEqual(b.strides, (s2.get_bytesize(), s.get_pitch()))
        self.assertTrue(b.suboffsets is None)
        self.assertFalse(b.readonly)
        self.assertEqual(b.buf, s2._pixels_address)
        b = Importer(a, buftools.PyBUF_RECORDS)
        self.assertEqual(b.ndim, 2)
        self.assertEqual(b.format, '=I')
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_SIMPLE)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_FORMAT)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_WRITABLE)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_ND)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_C_CONTIGUOUS)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_F_CONTIGUOUS)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_ANY_CONTIGUOUS)

    @unittest.skipIf(not pygame.HAVE_NEWBUF, 'newbuf not implemented')
    def test_newbuf_PyBUF_flags_3D(self):
        from pygame.tests.test_utils import buftools
        Importer = buftools.Importer
        s = pygame.Surface((12, 6), 0, 24)
        rmask, gmask, bmask, amask = s.get_masks()
        if self.lilendian:
            if rmask == 255:
                color_step = 1
                addr_offset = 0
            else:
                color_step = -1
                addr_offset = 2
        elif rmask == 16711680:
            color_step = 1
            addr_offset = 0
        else:
            color_step = -1
            addr_offset = 2
        a = s.get_view('3')
        b = Importer(a, buftools.PyBUF_STRIDES)
        w, h = s.get_size()
        shape = (w, h, 3)
        strides = (3, s.get_pitch(), color_step)
        self.assertEqual(b.ndim, 3)
        self.assertTrue(b.format is None)
        self.assertEqual(b.len, a.length)
        self.assertEqual(b.itemsize, 1)
        self.assertEqual(b.shape, shape)
        self.assertEqual(b.strides, strides)
        self.assertTrue(b.suboffsets is None)
        self.assertFalse(b.readonly)
        self.assertEqual(b.buf, s._pixels_address + addr_offset)
        b = Importer(a, buftools.PyBUF_RECORDS_RO)
        self.assertEqual(b.ndim, 3)
        self.assertEqual(b.format, 'B')
        self.assertEqual(b.strides, strides)
        b = Importer(a, buftools.PyBUF_RECORDS)
        self.assertEqual(b.ndim, 3)
        self.assertEqual(b.format, 'B')
        self.assertEqual(b.strides, strides)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_SIMPLE)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_FORMAT)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_WRITABLE)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_ND)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_C_CONTIGUOUS)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_F_CONTIGUOUS)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_ANY_CONTIGUOUS)

    @unittest.skipIf(not pygame.HAVE_NEWBUF, 'newbuf not implemented')
    def test_newbuf_PyBUF_flags_rgba(self):
        from pygame.tests.test_utils import buftools
        Importer = buftools.Importer
        s = pygame.Surface((12, 6), 0, 24)
        rmask, gmask, bmask, amask = s.get_masks()
        if self.lilendian:
            if rmask == 255:
                addr_offset = 0
            else:
                addr_offset = 2
        elif rmask == 16711680:
            addr_offset = 0
        else:
            addr_offset = 2
        a = s.get_view('R')
        b = Importer(a, buftools.PyBUF_STRIDES)
        w, h = s.get_size()
        shape = (w, h)
        strides = (s.get_bytesize(), s.get_pitch())
        self.assertEqual(b.ndim, 2)
        self.assertTrue(b.format is None)
        self.assertEqual(b.len, a.length)
        self.assertEqual(b.itemsize, 1)
        self.assertEqual(b.shape, shape)
        self.assertEqual(b.strides, strides)
        self.assertTrue(b.suboffsets is None)
        self.assertFalse(b.readonly)
        self.assertEqual(b.buf, s._pixels_address + addr_offset)
        b = Importer(a, buftools.PyBUF_RECORDS_RO)
        self.assertEqual(b.ndim, 2)
        self.assertEqual(b.format, 'B')
        self.assertEqual(b.strides, strides)
        b = Importer(a, buftools.PyBUF_RECORDS)
        self.assertEqual(b.ndim, 2)
        self.assertEqual(b.format, 'B')
        self.assertEqual(b.strides, strides)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_SIMPLE)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_FORMAT)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_WRITABLE)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_ND)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_C_CONTIGUOUS)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_F_CONTIGUOUS)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_ANY_CONTIGUOUS)