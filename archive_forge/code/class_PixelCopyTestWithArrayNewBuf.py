import platform
import unittest
import pygame
from pygame.locals import *
from pygame.pixelcopy import surface_to_array, map_array, array_to_surface, make_surface
@unittest.skipIf(not pygame.HAVE_NEWBUF, 'newbuf not implemented')
@unittest.skipIf(IS_PYPY, 'pypy having illegal instruction on mac')
class PixelCopyTestWithArrayNewBuf(unittest.TestCase):
    if pygame.HAVE_NEWBUF:
        from pygame.tests.test_utils import buftools

        class Array2D(buftools.Exporter):

            def __init__(self, initializer):
                from ctypes import cast, POINTER, c_uint32
                Array2D = PixelCopyTestWithArrayNewBuf.Array2D
                super().__init__((3, 5), format='=I', strides=(20, 4))
                self.content = cast(self.buf, POINTER(c_uint32))
                for i, v in enumerate(initializer):
                    self.content[i] = v

            def __getitem__(self, key):
                byte_index = key[0] * 5 + key[1]
                if not 0 <= byte_index < 15:
                    raise IndexError('%s is out of range', key)
                return self.content[byte_index]

        class Array3D(buftools.Exporter):

            def __init__(self, initializer):
                from ctypes import cast, POINTER, c_uint8
                Array3D = PixelCopyTestWithArrayNewBuf.Array3D
                super().__init__((3, 5, 3), format='B', strides=(20, 4, 1))
                self.content = cast(self.buf, POINTER(c_uint8))
                for i, v in enumerate(initializer):
                    self.content[i] = v

            def __getitem__(self, key):
                byte_index = key[0] * 20 + key[1] * 4 + key[2]
                if not 0 <= byte_index < 60:
                    raise IndexError('%s is out of range', key)
                return self.content[byte_index]
    surface = pygame.Surface((3, 5), 0, 32)

    def setUp(self):
        surf = self.surface
        for y in range(5):
            for x in range(3):
                surf.set_at((x, y), (x + 1, 0, y + 1))

    def assertCopy2D(self, surface, array):
        for x in range(0, 3):
            for y in range(0, 5):
                self.assertEqual(surface.get_at_mapped((x, y)), array[x, y])

    def test_surface_to_array_newbuf(self):
        array = self.Array2D(range(0, 15))
        self.assertNotEqual(array.content[0], self.surface.get_at_mapped((0, 0)))
        surface_to_array(array, self.surface)
        self.assertCopy2D(self.surface, array)

    def test_array_to_surface_newbuf(self):
        array = self.Array2D(range(0, 15))
        self.assertNotEqual(array.content[0], self.surface.get_at_mapped((0, 0)))
        array_to_surface(self.surface, array)
        self.assertCopy2D(self.surface, array)

    def test_map_array_newbuf(self):
        array2D = self.Array2D([0] * 15)
        elements = [i + (255 - i << 8) + (99 << 16) for i in range(0, 15)]
        array3D = self.Array3D(elements)
        map_array(array2D, array3D, self.surface)
        for x in range(0, 3):
            for y in range(0, 5):
                p = (array3D[x, y, 0], array3D[x, y, 1], array3D[x, y, 2])
                self.assertEqual(self.surface.unmap_rgb(array2D[x, y]), p)

    def test_make_surface_newbuf(self):
        array = self.Array2D(range(10, 160, 10))
        surface = make_surface(array)
        self.assertCopy2D(surface, array)

    def test_format_newbuf(self):
        Exporter = self.buftools.Exporter
        surface = self.surface
        shape = surface.get_size()
        w, h = shape
        for format in ['=i', '=I', '=l', '=L', '=q', '=Q', '<i', '>i', '!i', '1i', '=1i', '@q', 'q', '4x', '8x']:
            surface.fill((255, 254, 253))
            exp = Exporter(shape, format=format)
            exp._buf[:] = [42] * exp.buflen
            array_to_surface(surface, exp)
            for x in range(w):
                for y in range(h):
                    self.assertEqual(surface.get_at((x, y)), (42, 42, 42, 255))
        for format in ['f', 'd', '?', 'x', '1x', '2x', '3x', '5x', '6x', '7x', '9x']:
            exp = Exporter(shape, format=format)
            self.assertRaises(ValueError, array_to_surface, surface, exp)