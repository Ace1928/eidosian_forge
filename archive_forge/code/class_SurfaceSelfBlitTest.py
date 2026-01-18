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
class SurfaceSelfBlitTest(unittest.TestCase):
    """Blit to self tests.

    This test case is in response to https://github.com/pygame/pygame/issues/19
    """

    def setUp(self):
        pygame.display.init()

    def tearDown(self):
        pygame.display.quit()
    _test_palette = [(0, 0, 0, 255), (255, 0, 0, 0), (0, 255, 0, 255)]
    surf_size = (9, 6)

    def _fill_surface(self, surf, palette=None):
        if palette is None:
            palette = self._test_palette
        surf.fill(palette[1])
        surf.fill(palette[2], (1, 2, 1, 2))

    def _make_surface(self, bitsize, srcalpha=False, palette=None):
        if palette is None:
            palette = self._test_palette
        flags = 0
        if srcalpha:
            flags |= SRCALPHA
        surf = pygame.Surface(self.surf_size, flags, bitsize)
        if bitsize == 8:
            surf.set_palette([c[:3] for c in palette])
        self._fill_surface(surf, palette)
        return surf

    def _assert_same(self, a, b):
        w, h = a.get_size()
        for x in range(w):
            for y in range(h):
                self.assertEqual(a.get_at((x, y)), b.get_at((x, y)), '%s != %s, bpp: %i' % (a.get_at((x, y)), b.get_at((x, y)), a.get_bitsize()))

    def test_overlap_check(self):
        bgc = (0, 0, 0, 255)
        rectc_left = (128, 64, 32, 255)
        rectc_right = (255, 255, 255, 255)
        colors = [(255, 255, 255, 255), (128, 64, 32, 255)]
        overlaps = [(0, 0, 1, 0, (50, 0)), (0, 0, 49, 1, (98, 2)), (0, 0, 49, 49, (98, 98)), (49, 0, 0, 1, (0, 2)), (49, 0, 0, 49, (0, 98))]
        surfs = [pygame.Surface((100, 100), SRCALPHA, 32)]
        surf = pygame.Surface((100, 100), 0, 32)
        surf.set_alpha(255)
        surfs.append(surf)
        surf = pygame.Surface((100, 100), 0, 32)
        surf.set_colorkey((0, 1, 0))
        surfs.append(surf)
        for surf in surfs:
            for s_x, s_y, d_x, d_y, test_posn in overlaps:
                surf.fill(bgc)
                surf.fill(rectc_right, (25, 0, 25, 50))
                surf.fill(rectc_left, (0, 0, 25, 50))
                surf.blit(surf, (d_x, d_y), (s_x, s_y, 50, 50))
                self.assertEqual(surf.get_at(test_posn), rectc_right)

    @unittest.skipIf('ppc64le' in platform.uname(), 'known ppc64le issue')
    def test_colorkey(self):
        pygame.display.set_mode((100, 50))
        bitsizes = [8, 16, 24, 32]
        for bitsize in bitsizes:
            surf = self._make_surface(bitsize)
            surf.set_colorkey(self._test_palette[1])
            surf.blit(surf, (3, 0))
            p = []
            for c in self._test_palette:
                c = surf.unmap_rgb(surf.map_rgb(c))
                p.append(c)
            p[1] = (p[1][0], p[1][1], p[1][2], 0)
            tmp = self._make_surface(32, srcalpha=True, palette=p)
            tmp.blit(tmp, (3, 0))
            tmp.set_alpha(None)
            comp = self._make_surface(bitsize)
            comp.blit(tmp, (0, 0))
            self._assert_same(surf, comp)

    @unittest.skipIf('ppc64le' in platform.uname(), 'known ppc64le issue')
    def test_blanket_alpha(self):
        pygame.display.set_mode((100, 50))
        bitsizes = [8, 16, 24, 32]
        for bitsize in bitsizes:
            surf = self._make_surface(bitsize)
            surf.set_alpha(128)
            surf.blit(surf, (3, 0))
            p = []
            for c in self._test_palette:
                c = surf.unmap_rgb(surf.map_rgb(c))
                p.append((c[0], c[1], c[2], 128))
            tmp = self._make_surface(32, srcalpha=True, palette=p)
            tmp.blit(tmp, (3, 0))
            tmp.set_alpha(None)
            comp = self._make_surface(bitsize)
            comp.blit(tmp, (0, 0))
            self._assert_same(surf, comp)

    def test_pixel_alpha(self):
        bitsizes = [16, 32]
        for bitsize in bitsizes:
            surf = self._make_surface(bitsize, srcalpha=True)
            comp = self._make_surface(bitsize, srcalpha=True)
            comp.blit(surf, (3, 0))
            surf.blit(surf, (3, 0))
            self._assert_same(surf, comp)

    def test_blend(self):
        bitsizes = [8, 16, 24, 32]
        blends = ['BLEND_ADD', 'BLEND_SUB', 'BLEND_MULT', 'BLEND_MIN', 'BLEND_MAX']
        for bitsize in bitsizes:
            surf = self._make_surface(bitsize)
            comp = self._make_surface(bitsize)
            for blend in blends:
                self._fill_surface(surf)
                self._fill_surface(comp)
                comp.blit(surf, (3, 0), special_flags=getattr(pygame, blend))
                surf.blit(surf, (3, 0), special_flags=getattr(pygame, blend))
                self._assert_same(surf, comp)

    def test_blend_rgba(self):
        bitsizes = [16, 32]
        blends = ['BLEND_RGBA_ADD', 'BLEND_RGBA_SUB', 'BLEND_RGBA_MULT', 'BLEND_RGBA_MIN', 'BLEND_RGBA_MAX']
        for bitsize in bitsizes:
            surf = self._make_surface(bitsize, srcalpha=True)
            comp = self._make_surface(bitsize, srcalpha=True)
            for blend in blends:
                self._fill_surface(surf)
                self._fill_surface(comp)
                comp.blit(surf, (3, 0), special_flags=getattr(pygame, blend))
                surf.blit(surf, (3, 0), special_flags=getattr(pygame, blend))
                self._assert_same(surf, comp)

    def test_subsurface(self):
        surf = self._make_surface(32, srcalpha=True)
        comp = surf.copy()
        comp.blit(surf, (3, 0))
        sub = surf.subsurface((3, 0, 6, 6))
        sub.blit(surf, (0, 0))
        del sub
        self._assert_same(surf, comp)

        def do_blit(d, s):
            d.blit(s, (0, 0))
        sub = surf.subsurface((1, 1, 2, 2))
        self.assertRaises(pygame.error, do_blit, surf, sub)

    def test_copy_alpha(self):
        """issue 581: alpha of surface copy with SRCALPHA is set to 0."""
        surf = pygame.Surface((16, 16), pygame.SRCALPHA, 32)
        self.assertEqual(surf.get_alpha(), 255)
        surf2 = surf.copy()
        self.assertEqual(surf2.get_alpha(), 255)