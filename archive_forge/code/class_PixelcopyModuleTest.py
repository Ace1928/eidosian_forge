import platform
import unittest
import pygame
from pygame.locals import *
from pygame.pixelcopy import surface_to_array, map_array, array_to_surface, make_surface
@unittest.skipIf(IS_PYPY, 'pypy having illegal instruction on mac')
class PixelcopyModuleTest(unittest.TestCase):
    bitsizes = [8, 16, 32]
    test_palette = [(0, 0, 0, 255), (10, 30, 60, 255), (25, 75, 100, 255), (100, 150, 200, 255), (0, 100, 200, 255)]
    surf_size = (10, 12)
    test_points = [((0, 0), 1), ((4, 5), 1), ((9, 0), 2), ((5, 5), 2), ((0, 11), 3), ((4, 6), 3), ((9, 11), 4), ((5, 6), 4)]

    def __init__(self, *args, **kwds):
        pygame.display.init()
        try:
            unittest.TestCase.__init__(self, *args, **kwds)
            self.sources = [self._make_src_surface(8), self._make_src_surface(16), self._make_src_surface(16, srcalpha=True), self._make_src_surface(24), self._make_src_surface(32), self._make_src_surface(32, srcalpha=True)]
        finally:
            pygame.display.quit()

    def _make_surface(self, bitsize, srcalpha=False, palette=None):
        if palette is None:
            palette = self.test_palette
        flags = 0
        if srcalpha:
            flags |= SRCALPHA
        surf = pygame.Surface(self.surf_size, flags, bitsize)
        if bitsize == 8:
            surf.set_palette([c[:3] for c in palette])
        return surf

    def _fill_surface(self, surf, palette=None):
        if palette is None:
            palette = self.test_palette
        surf.fill(palette[1], (0, 0, 5, 6))
        surf.fill(palette[2], (5, 0, 5, 6))
        surf.fill(palette[3], (0, 6, 5, 6))
        surf.fill(palette[4], (5, 6, 5, 6))

    def _make_src_surface(self, bitsize, srcalpha=False, palette=None):
        surf = self._make_surface(bitsize, srcalpha, palette)
        self._fill_surface(surf, palette)
        return surf

    def setUp(self):
        pygame.display.init()

    def tearDown(self):
        pygame.display.quit()

    def test_surface_to_array_2d(self):
        alpha_color = (0, 0, 0, 128)
        for surf in self.sources:
            src_bitsize = surf.get_bitsize()
            for dst_bitsize in self.bitsizes:
                dst = pygame.Surface(surf.get_size(), 0, dst_bitsize)
                dst.fill((0, 0, 0, 0))
                view = dst.get_view('2')
                self.assertFalse(surf.get_locked())
                if dst_bitsize < src_bitsize:
                    self.assertRaises(ValueError, surface_to_array, view, surf)
                    self.assertFalse(surf.get_locked())
                    continue
                surface_to_array(view, surf)
                self.assertFalse(surf.get_locked())
                for posn, i in self.test_points:
                    sp = surf.get_at_mapped(posn)
                    dp = dst.get_at_mapped(posn)
                    self.assertEqual(dp, sp, '%s != %s: flags: %i, bpp: %i, posn: %s' % (dp, sp, surf.get_flags(), surf.get_bitsize(), posn))
                del view
                if surf.get_masks()[3]:
                    dst.fill((0, 0, 0, 0))
                    view = dst.get_view('2')
                    posn = (2, 1)
                    surf.set_at(posn, alpha_color)
                    self.assertFalse(surf.get_locked())
                    surface_to_array(view, surf)
                    self.assertFalse(surf.get_locked())
                    sp = surf.get_at_mapped(posn)
                    dp = dst.get_at_mapped(posn)
                    self.assertEqual(dp, sp, '%s != %s: bpp: %i' % (dp, sp, surf.get_bitsize()))
        if IS_PYPY:
            return
        pai_flags = arrinter.PAI_ALIGNED | arrinter.PAI_WRITEABLE
        for surf in self.sources:
            for itemsize in [1, 2, 4, 8]:
                if itemsize < surf.get_bytesize():
                    continue
                a = arrinter.Array(surf.get_size(), 'u', itemsize, flags=pai_flags)
                surface_to_array(a, surf)
                for posn, i in self.test_points:
                    sp = unsigned32(surf.get_at_mapped(posn))
                    dp = a[posn]
                    self.assertEqual(dp, sp, '%s != %s: itemsize: %i, flags: %i, bpp: %i, posn: %s' % (dp, sp, itemsize, surf.get_flags(), surf.get_bitsize(), posn))

    def test_surface_to_array_3d(self):
        self.iter_surface_to_array_3d((255, 65280, 16711680, 0))
        self.iter_surface_to_array_3d((16711680, 65280, 255, 0))

    def iter_surface_to_array_3d(self, rgba_masks):
        dst = pygame.Surface(self.surf_size, 0, 24, masks=rgba_masks)
        for surf in self.sources:
            dst.fill((0, 0, 0, 0))
            src_bitsize = surf.get_bitsize()
            view = dst.get_view('3')
            self.assertFalse(surf.get_locked())
            surface_to_array(view, surf)
            self.assertFalse(surf.get_locked())
            for posn, i in self.test_points:
                sc = surf.get_at(posn)[0:3]
                dc = dst.get_at(posn)[0:3]
                self.assertEqual(dc, sc, '%s != %s: flags: %i, bpp: %i, posn: %s' % (dc, sc, surf.get_flags(), surf.get_bitsize(), posn))
            view = None

    def test_map_array(self):
        targets = [self._make_surface(8), self._make_surface(16), self._make_surface(16, srcalpha=True), self._make_surface(24), self._make_surface(32), self._make_surface(32, srcalpha=True)]
        source = pygame.Surface(self.surf_size, 0, 24, masks=[255, 65280, 16711680, 0])
        self._fill_surface(source)
        source_view = source.get_view('3')
        for t in targets:
            map_array(t.get_view('2'), source_view, t)
            for posn, i in self.test_points:
                sc = t.map_rgb(source.get_at(posn))
                dc = t.get_at_mapped(posn)
                self.assertEqual(dc, sc, '%s != %s: flags: %i, bpp: %i, posn: %s' % (dc, sc, t.get_flags(), t.get_bitsize(), posn))
        color = pygame.Color('salmon')
        color.set_length(3)
        for t in targets:
            map_array(t.get_view('2'), color, t)
            sc = t.map_rgb(color)
            for posn, i in self.test_points:
                dc = t.get_at_mapped(posn)
                self.assertEqual(dc, sc, '%s != %s: flags: %i, bpp: %i, posn: %s' % (dc, sc, t.get_flags(), t.get_bitsize(), posn))
        w, h = source.get_size()
        target = pygame.Surface((w, h + 1), 0, 32)
        self.assertRaises(ValueError, map_array, target, source, target)
        target = pygame.Surface((w - 1, h), 0, 32)
        self.assertRaises(ValueError, map_array, target, source, target)

    def test_array_to_surface_broadcasting(self):
        targets = [self._make_surface(8), self._make_surface(16), self._make_surface(16, srcalpha=True), self._make_surface(24), self._make_surface(32), self._make_surface(32, srcalpha=True)]
        w, h = self.surf_size
        column = pygame.Surface((1, h), 0, 32)
        for target in targets:
            source = pygame.Surface((1, h), 0, target)
            for y in range(h):
                source.set_at((0, y), pygame.Color(y + 1, y + h + 1, y + 2 * h + 1))
            pygame.pixelcopy.surface_to_array(column.get_view('2'), source)
            pygame.pixelcopy.array_to_surface(target, column.get_view('2'))
            for x in range(w):
                for y in range(h):
                    self.assertEqual(target.get_at_mapped((x, y)), column.get_at_mapped((0, y)))
        row = pygame.Surface((w, 1), 0, 32)
        for target in targets:
            source = pygame.Surface((w, 1), 0, target)
            for x in range(w):
                source.set_at((x, 0), pygame.Color(x + 1, x + w + 1, x + 2 * w + 1))
            pygame.pixelcopy.surface_to_array(row.get_view('2'), source)
            pygame.pixelcopy.array_to_surface(target, row.get_view('2'))
            for x in range(w):
                for y in range(h):
                    self.assertEqual(target.get_at_mapped((x, y)), row.get_at_mapped((x, 0)))
        pixel = pygame.Surface((1, 1), 0, 32)
        for target in targets:
            source = pygame.Surface((1, 1), 0, target)
            source.set_at((0, 0), pygame.Color(13, 47, 101))
            pygame.pixelcopy.surface_to_array(pixel.get_view('2'), source)
            pygame.pixelcopy.array_to_surface(target, pixel.get_view('2'))
            p = pixel.get_at_mapped((0, 0))
            for x in range(w):
                for y in range(h):
                    self.assertEqual(target.get_at_mapped((x, y)), p)