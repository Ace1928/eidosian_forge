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
class SurfaceBlendTest(unittest.TestCase):

    def setUp(self):
        pygame.display.init()

    def tearDown(self):
        pygame.display.quit()
    _test_palette = [(0, 0, 0, 255), (10, 30, 60, 0), (25, 75, 100, 128), (200, 150, 100, 200), (0, 100, 200, 255)]
    surf_size = (10, 12)
    _test_points = [((0, 0), 1), ((4, 5), 1), ((9, 0), 2), ((5, 5), 2), ((0, 11), 3), ((4, 6), 3), ((9, 11), 4), ((5, 6), 4)]

    def _make_surface(self, bitsize, srcalpha=False, palette=None):
        if palette is None:
            palette = self._test_palette
        flags = 0
        if srcalpha:
            flags |= SRCALPHA
        surf = pygame.Surface(self.surf_size, flags, bitsize)
        if bitsize == 8:
            surf.set_palette([c[:3] for c in palette])
        return surf

    def _fill_surface(self, surf, palette=None):
        if palette is None:
            palette = self._test_palette
        surf.fill(palette[1], (0, 0, 5, 6))
        surf.fill(palette[2], (5, 0, 5, 6))
        surf.fill(palette[3], (0, 6, 5, 6))
        surf.fill(palette[4], (5, 6, 5, 6))

    def _make_src_surface(self, bitsize, srcalpha=False, palette=None):
        surf = self._make_surface(bitsize, srcalpha, palette)
        self._fill_surface(surf, palette)
        return surf

    def _assert_surface(self, surf, palette=None, msg=''):
        if palette is None:
            palette = self._test_palette
        if surf.get_bitsize() == 16:
            palette = [surf.unmap_rgb(surf.map_rgb(c)) for c in palette]
        for posn, i in self._test_points:
            self.assertEqual(surf.get_at(posn), palette[i], '%s != %s: flags: %i, bpp: %i, posn: %s%s' % (surf.get_at(posn), palette[i], surf.get_flags(), surf.get_bitsize(), posn, msg))

    def test_blit_blend(self):
        sources = [self._make_src_surface(8), self._make_src_surface(16), self._make_src_surface(16, srcalpha=True), self._make_src_surface(24), self._make_src_surface(32), self._make_src_surface(32, srcalpha=True)]
        destinations = [self._make_surface(8), self._make_surface(16), self._make_surface(16, srcalpha=True), self._make_surface(24), self._make_surface(32), self._make_surface(32, srcalpha=True)]
        blend = [('BLEND_ADD', (0, 25, 100, 255), lambda a, b: min(a + b, 255)), ('BLEND_SUB', (100, 25, 0, 100), lambda a, b: max(a - b, 0)), ('BLEND_MULT', (100, 200, 0, 0), lambda a, b: a * b + 255 >> 8), ('BLEND_MIN', (255, 0, 0, 255), min), ('BLEND_MAX', (0, 255, 0, 255), max)]
        for src in sources:
            src_palette = [src.unmap_rgb(src.map_rgb(c)) for c in self._test_palette]
            for dst in destinations:
                for blend_name, dst_color, op in blend:
                    dc = dst.unmap_rgb(dst.map_rgb(dst_color))
                    p = []
                    for sc in src_palette:
                        c = [op(dc[i], sc[i]) for i in range(3)]
                        if dst.get_masks()[3]:
                            c.append(dc[3])
                        else:
                            c.append(255)
                        c = dst.unmap_rgb(dst.map_rgb(c))
                        p.append(c)
                    dst.fill(dst_color)
                    dst.blit(src, (0, 0), special_flags=getattr(pygame, blend_name))
                    self._assert_surface(dst, p, ', op: %s, src bpp: %i, src flags: %i' % (blend_name, src.get_bitsize(), src.get_flags()))
        src = self._make_src_surface(32)
        masks = src.get_masks()
        dst = pygame.Surface(src.get_size(), 0, 32, [masks[2], masks[1], masks[0], masks[3]])
        for blend_name, dst_color, op in blend:
            p = []
            for src_color in self._test_palette:
                c = [op(dst_color[i], src_color[i]) for i in range(3)]
                c.append(255)
                p.append(tuple(c))
            dst.fill(dst_color)
            dst.blit(src, (0, 0), special_flags=getattr(pygame, blend_name))
            self._assert_surface(dst, p, f', {blend_name}')
        pat = self._make_src_surface(32)
        masks = pat.get_masks()
        if min(masks) == 4278190080:
            masks = [m >> 8 for m in masks]
        else:
            masks = [m << 8 for m in masks]
        src = pygame.Surface(pat.get_size(), 0, 32, masks)
        self._fill_surface(src)
        dst = pygame.Surface(src.get_size(), 0, 32, masks)
        for blend_name, dst_color, op in blend:
            p = []
            for src_color in self._test_palette:
                c = [op(dst_color[i], src_color[i]) for i in range(3)]
                c.append(255)
                p.append(tuple(c))
            dst.fill(dst_color)
            dst.blit(src, (0, 0), special_flags=getattr(pygame, blend_name))
            self._assert_surface(dst, p, f', {blend_name}')

    def test_blit_blend_rgba(self):
        sources = [self._make_src_surface(8), self._make_src_surface(16), self._make_src_surface(16, srcalpha=True), self._make_src_surface(24), self._make_src_surface(32), self._make_src_surface(32, srcalpha=True)]
        destinations = [self._make_surface(8), self._make_surface(16), self._make_surface(16, srcalpha=True), self._make_surface(24), self._make_surface(32), self._make_surface(32, srcalpha=True)]
        blend = [('BLEND_RGBA_ADD', (0, 25, 100, 255), lambda a, b: min(a + b, 255)), ('BLEND_RGBA_SUB', (0, 25, 100, 255), lambda a, b: max(a - b, 0)), ('BLEND_RGBA_MULT', (0, 7, 100, 255), lambda a, b: a * b + 255 >> 8), ('BLEND_RGBA_MIN', (0, 255, 0, 255), min), ('BLEND_RGBA_MAX', (0, 255, 0, 255), max)]
        for src in sources:
            src_palette = [src.unmap_rgb(src.map_rgb(c)) for c in self._test_palette]
            for dst in destinations:
                for blend_name, dst_color, op in blend:
                    dc = dst.unmap_rgb(dst.map_rgb(dst_color))
                    p = []
                    for sc in src_palette:
                        c = [op(dc[i], sc[i]) for i in range(4)]
                        if not dst.get_masks()[3]:
                            c[3] = 255
                        c = dst.unmap_rgb(dst.map_rgb(c))
                        p.append(c)
                    dst.fill(dst_color)
                    dst.blit(src, (0, 0), special_flags=getattr(pygame, blend_name))
                    self._assert_surface(dst, p, ', op: %s, src bpp: %i, src flags: %i' % (blend_name, src.get_bitsize(), src.get_flags()))
        src = self._make_src_surface(32, srcalpha=True)
        masks = src.get_masks()
        dst = pygame.Surface(src.get_size(), SRCALPHA, 32, (masks[2], masks[1], masks[0], masks[3]))
        for blend_name, dst_color, op in blend:
            p = [tuple((op(dst_color[i], src_color[i]) for i in range(4))) for src_color in self._test_palette]
            dst.fill(dst_color)
            dst.blit(src, (0, 0), special_flags=getattr(pygame, blend_name))
            self._assert_surface(dst, p, f', {blend_name}')
        src = pygame.Surface((8, 10), SRCALPHA, 32)
        dst = pygame.Surface((8, 10), SRCALPHA, 32)
        tst = pygame.Surface((8, 10), SRCALPHA, 32)
        src.fill((1, 2, 3, 4))
        dst.fill((40, 30, 20, 10))
        subsrc = src.subsurface((2, 3, 4, 4))
        subdst = dst.subsurface((2, 3, 4, 4))
        subdst.blit(subsrc, (0, 0), special_flags=BLEND_RGBA_ADD)
        tst.fill((40, 30, 20, 10))
        tst.fill((41, 32, 23, 14), (2, 3, 4, 4))
        for x in range(8):
            for y in range(10):
                self.assertEqual(dst.get_at((x, y)), tst.get_at((x, y)), '%s != %s at (%i, %i)' % (dst.get_at((x, y)), tst.get_at((x, y)), x, y))

    def test_blit_blend_premultiplied(self):

        def test_premul_surf(src_col, dst_col, src_size=(16, 16), dst_size=(16, 16), src_bit_depth=32, dst_bit_depth=32, src_has_alpha=True, dst_has_alpha=True):
            if src_bit_depth == 8:
                src = pygame.Surface(src_size, 0, src_bit_depth)
                palette = [src_col, dst_col]
                src.set_palette(palette)
                src.fill(palette[0])
            elif src_has_alpha:
                src = pygame.Surface(src_size, SRCALPHA, src_bit_depth)
                src.fill(src_col)
            else:
                src = pygame.Surface(src_size, 0, src_bit_depth)
                src.fill(src_col)
            if dst_bit_depth == 8:
                dst = pygame.Surface(dst_size, 0, dst_bit_depth)
                palette = [src_col, dst_col]
                dst.set_palette(palette)
                dst.fill(palette[1])
            elif dst_has_alpha:
                dst = pygame.Surface(dst_size, SRCALPHA, dst_bit_depth)
                dst.fill(dst_col)
            else:
                dst = pygame.Surface(dst_size, 0, dst_bit_depth)
                dst.fill(dst_col)
            dst.blit(src, (0, 0), special_flags=BLEND_PREMULTIPLIED)
            actual_col = dst.get_at((int(float(src_size[0] / 2.0)), int(float(src_size[0] / 2.0))))
            if src_col.a == 0:
                expected_col = dst_col
            elif src_col.a == 255:
                expected_col = src_col
            else:
                expected_col = pygame.Color(src_col.r + dst_col.r - ((dst_col.r + 1) * src_col.a >> 8), src_col.g + dst_col.g - ((dst_col.g + 1) * src_col.a >> 8), src_col.b + dst_col.b - ((dst_col.b + 1) * src_col.a >> 8), src_col.a + dst_col.a - ((dst_col.a + 1) * src_col.a >> 8))
            if not dst_has_alpha:
                expected_col.a = 255
            return (expected_col, actual_col)
        self.assertEqual(*test_premul_surf(pygame.Color(40, 20, 0, 51), pygame.Color(40, 20, 0, 51)))
        self.assertEqual(*test_premul_surf(pygame.Color(0, 0, 0, 0), pygame.Color(40, 20, 0, 51)))
        self.assertEqual(*test_premul_surf(pygame.Color(40, 20, 0, 51), pygame.Color(0, 0, 0, 0)))
        self.assertEqual(*test_premul_surf(pygame.Color(0, 0, 0, 0), pygame.Color(0, 0, 0, 0)))
        self.assertEqual(*test_premul_surf(pygame.Color(2, 2, 2, 2), pygame.Color(40, 20, 0, 51)))
        self.assertEqual(*test_premul_surf(pygame.Color(40, 20, 0, 51), pygame.Color(2, 2, 2, 2)))
        self.assertEqual(*test_premul_surf(pygame.Color(2, 2, 2, 2), pygame.Color(2, 2, 2, 2)))
        self.assertEqual(*test_premul_surf(pygame.Color(9, 9, 9, 9), pygame.Color(40, 20, 0, 51)))
        self.assertEqual(*test_premul_surf(pygame.Color(40, 20, 0, 51), pygame.Color(9, 9, 9, 9)))
        self.assertEqual(*test_premul_surf(pygame.Color(9, 9, 9, 9), pygame.Color(9, 9, 9, 9)))
        self.assertEqual(*test_premul_surf(pygame.Color(127, 127, 127, 127), pygame.Color(40, 20, 0, 51)))
        self.assertEqual(*test_premul_surf(pygame.Color(40, 20, 0, 51), pygame.Color(127, 127, 127, 127)))
        self.assertEqual(*test_premul_surf(pygame.Color(127, 127, 127, 127), pygame.Color(127, 127, 127, 127)))
        self.assertEqual(*test_premul_surf(pygame.Color(200, 200, 200, 200), pygame.Color(40, 20, 0, 51)))
        self.assertEqual(*test_premul_surf(pygame.Color(40, 20, 0, 51), pygame.Color(200, 200, 200, 200)))
        self.assertEqual(*test_premul_surf(pygame.Color(200, 200, 200, 200), pygame.Color(200, 200, 200, 200)))
        self.assertEqual(*test_premul_surf(pygame.Color(255, 255, 255, 255), pygame.Color(40, 20, 0, 51)))
        self.assertEqual(*test_premul_surf(pygame.Color(40, 20, 0, 51), pygame.Color(255, 255, 255, 255)))
        self.assertEqual(*test_premul_surf(pygame.Color(255, 255, 255, 255), pygame.Color(255, 255, 255, 255)))
        self.assertRaises(IndexError, test_premul_surf, pygame.Color(255, 255, 255, 255), pygame.Color(255, 255, 255, 255), src_size=(0, 0), dst_size=(0, 0))
        self.assertEqual(*test_premul_surf(pygame.Color(40, 20, 0, 51), pygame.Color(30, 20, 0, 51), src_size=(4, 4), dst_size=(9, 9)))
        self.assertEqual(*test_premul_surf(pygame.Color(30, 20, 0, 51), pygame.Color(40, 20, 0, 51), src_size=(17, 67), dst_size=(69, 69)))
        self.assertEqual(*test_premul_surf(pygame.Color(30, 20, 0, 255), pygame.Color(40, 20, 0, 51), src_size=(17, 67), dst_size=(69, 69), src_has_alpha=True))
        self.assertEqual(*test_premul_surf(pygame.Color(30, 20, 0, 51), pygame.Color(40, 20, 0, 255), src_size=(17, 67), dst_size=(69, 69), dst_has_alpha=False))
        self.assertEqual(*test_premul_surf(pygame.Color(30, 20, 0, 255), pygame.Color(40, 20, 0, 255), src_size=(17, 67), dst_size=(69, 69), src_has_alpha=False, dst_has_alpha=False))
        self.assertEqual(*test_premul_surf(pygame.Color(30, 20, 0, 255), pygame.Color(40, 20, 0, 255), src_size=(17, 67), dst_size=(69, 69), dst_bit_depth=24, src_has_alpha=True, dst_has_alpha=False))
        self.assertEqual(*test_premul_surf(pygame.Color(30, 20, 0, 255), pygame.Color(40, 20, 0, 255), src_size=(17, 67), dst_size=(69, 69), src_bit_depth=24, src_has_alpha=False, dst_has_alpha=True))
        self.assertEqual(*test_premul_surf(pygame.Color(30, 20, 0, 255), pygame.Color(40, 20, 0, 255), src_size=(17, 67), dst_size=(69, 69), src_bit_depth=24, dst_bit_depth=24, src_has_alpha=False, dst_has_alpha=False))
        self.assertEqual(*test_premul_surf(pygame.Color(30, 20, 0, 255), pygame.Color(40, 20, 0, 255), src_size=(17, 67), dst_size=(69, 69), src_bit_depth=8))
        self.assertEqual(*test_premul_surf(pygame.Color(30, 20, 0, 255), pygame.Color(40, 20, 0, 255), src_size=(17, 67), dst_size=(69, 69), dst_bit_depth=8))
        self.assertEqual(*test_premul_surf(pygame.Color(30, 20, 0, 255), pygame.Color(40, 20, 0, 255), src_size=(17, 67), dst_size=(69, 69), src_bit_depth=8, dst_bit_depth=8))

    def test_blit_blend_big_rect(self):
        """test that an oversized rect works ok."""
        color = (1, 2, 3, 255)
        area = (1, 1, 30, 30)
        s1 = pygame.Surface((4, 4), 0, 32)
        r = s1.fill(special_flags=pygame.BLEND_ADD, color=color, rect=area)
        self.assertEqual(pygame.Rect((1, 1, 3, 3)), r)
        self.assertEqual(s1.get_at((0, 0)), (0, 0, 0, 255))
        self.assertEqual(s1.get_at((1, 1)), color)
        black = pygame.Color('black')
        red = pygame.Color('red')
        self.assertNotEqual(black, red)
        surf = pygame.Surface((10, 10), 0, 32)
        surf.fill(black)
        subsurf = surf.subsurface(pygame.Rect(0, 1, 10, 8))
        self.assertEqual(surf.get_at((0, 0)), black)
        self.assertEqual(surf.get_at((0, 9)), black)
        subsurf.fill(red, (0, -1, 10, 1), pygame.BLEND_RGB_ADD)
        self.assertEqual(surf.get_at((0, 0)), black)
        self.assertEqual(surf.get_at((0, 9)), black)
        subsurf.fill(red, (0, 8, 10, 1), pygame.BLEND_RGB_ADD)
        self.assertEqual(surf.get_at((0, 0)), black)
        self.assertEqual(surf.get_at((0, 9)), black)

    def test_GET_PIXELVALS(self):
        src = self._make_surface(32, srcalpha=True)
        src.fill((0, 0, 0, 128))
        src.set_alpha(None)
        dst = self._make_surface(32, srcalpha=True)
        dst.blit(src, (0, 0), special_flags=BLEND_RGBA_ADD)
        self.assertEqual(dst.get_at((0, 0)), (0, 0, 0, 255))

    def test_fill_blend(self):
        destinations = [self._make_surface(8), self._make_surface(16), self._make_surface(16, srcalpha=True), self._make_surface(24), self._make_surface(32), self._make_surface(32, srcalpha=True)]
        blend = [('BLEND_ADD', (0, 25, 100, 255), lambda a, b: min(a + b, 255)), ('BLEND_SUB', (0, 25, 100, 255), lambda a, b: max(a - b, 0)), ('BLEND_MULT', (0, 7, 100, 255), lambda a, b: a * b + 255 >> 8), ('BLEND_MIN', (0, 255, 0, 255), min), ('BLEND_MAX', (0, 255, 0, 255), max)]
        for dst in destinations:
            dst_palette = [dst.unmap_rgb(dst.map_rgb(c)) for c in self._test_palette]
            for blend_name, fill_color, op in blend:
                fc = dst.unmap_rgb(dst.map_rgb(fill_color))
                self._fill_surface(dst)
                p = []
                for dc in dst_palette:
                    c = [op(dc[i], fc[i]) for i in range(3)]
                    if dst.get_masks()[3]:
                        c.append(dc[3])
                    else:
                        c.append(255)
                    c = dst.unmap_rgb(dst.map_rgb(c))
                    p.append(c)
                dst.fill(fill_color, special_flags=getattr(pygame, blend_name))
                self._assert_surface(dst, p, f', {blend_name}')

    def test_fill_blend_rgba(self):
        destinations = [self._make_surface(8), self._make_surface(16), self._make_surface(16, srcalpha=True), self._make_surface(24), self._make_surface(32), self._make_surface(32, srcalpha=True)]
        blend = [('BLEND_RGBA_ADD', (0, 25, 100, 255), lambda a, b: min(a + b, 255)), ('BLEND_RGBA_SUB', (0, 25, 100, 255), lambda a, b: max(a - b, 0)), ('BLEND_RGBA_MULT', (0, 7, 100, 255), lambda a, b: a * b + 255 >> 8), ('BLEND_RGBA_MIN', (0, 255, 0, 255), min), ('BLEND_RGBA_MAX', (0, 255, 0, 255), max)]
        for dst in destinations:
            dst_palette = [dst.unmap_rgb(dst.map_rgb(c)) for c in self._test_palette]
            for blend_name, fill_color, op in blend:
                fc = dst.unmap_rgb(dst.map_rgb(fill_color))
                self._fill_surface(dst)
                p = []
                for dc in dst_palette:
                    c = [op(dc[i], fc[i]) for i in range(4)]
                    if not dst.get_masks()[3]:
                        c[3] = 255
                    c = dst.unmap_rgb(dst.map_rgb(c))
                    p.append(c)
                dst.fill(fill_color, special_flags=getattr(pygame, blend_name))
                self._assert_surface(dst, p, f', {blend_name}')

    def test_surface_premul_alpha(self):
        """Ensure that .premul_alpha() works correctly"""
        s1 = pygame.Surface((100, 100), pygame.SRCALPHA, 32)
        s1.fill(pygame.Color(255, 255, 255, 100))
        s1_alpha = s1.premul_alpha()
        self.assertEqual(s1_alpha.get_at((50, 50)), pygame.Color(100, 100, 100, 100))
        s2 = pygame.Surface((100, 100), pygame.SRCALPHA, 16)
        s2.fill(pygame.Color(int(15 / 15 * 255), int(15 / 15 * 255), int(15 / 15 * 255), int(10 / 15 * 255)))
        s2_alpha = s2.premul_alpha()
        self.assertEqual(s2_alpha.get_at((50, 50)), pygame.Color(int(10 / 15 * 255), int(10 / 15 * 255), int(10 / 15 * 255), int(10 / 15 * 255)))
        invalid_surf = pygame.Surface((100, 100), 0, 32)
        invalid_surf.fill(pygame.Color(255, 255, 255, 100))
        with self.assertRaises(ValueError):
            invalid_surf.premul_alpha()
        test_colors = [(200, 30, 74), (76, 83, 24), (184, 21, 6), (74, 4, 74), (76, 83, 24), (184, 21, 234), (160, 30, 74), (96, 147, 204), (198, 201, 60), (132, 89, 74), (245, 9, 224), (184, 112, 6)]
        for r, g, b in test_colors:
            for a in range(255):
                with self.subTest(r=r, g=g, b=b, a=a):
                    surf = pygame.Surface((10, 10), pygame.SRCALPHA, 32)
                    surf.fill(pygame.Color(r, g, b, a))
                    surf = surf.premul_alpha()
                    self.assertEqual(surf.get_at((5, 5)), Color((r + 1) * a >> 8, (g + 1) * a >> 8, (b + 1) * a >> 8, a))