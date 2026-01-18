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
class TestSurfaceBlit(unittest.TestCase):
    """Tests basic blitting functionality and options."""

    def setUp(self):
        """Resets starting surfaces."""
        self.src_surface = pygame.Surface((256, 256), 32)
        self.src_surface.fill(pygame.Color(255, 255, 255))
        self.dst_surface = pygame.Surface((64, 64), 32)
        self.dst_surface.fill(pygame.Color(0, 0, 0))

    def test_blit_overflow_coord(self):
        """Full coverage w/ overflow, specified with Coordinate"""
        result = self.dst_surface.blit(self.src_surface, (0, 0))
        self.assertIsInstance(result, pygame.Rect)
        self.assertEqual(result.size, (64, 64))
        for k in [(x, x) for x in range(64)]:
            self.assertEqual(self.dst_surface.get_at(k), (255, 255, 255))

    def test_blit_overflow_rect(self):
        """Full coverage w/ overflow, specified with a Rect"""
        result = self.dst_surface.blit(self.src_surface, pygame.Rect(-1, -1, 300, 300))
        self.assertIsInstance(result, pygame.Rect)
        self.assertEqual(result.size, (64, 64))
        for k in [(x, x) for x in range(64)]:
            self.assertEqual(self.dst_surface.get_at(k), (255, 255, 255))

    def test_blit_overflow_nonorigin(self):
        """Test Rectangle Dest, with overflow but with starting rect with top-left at (1,1)"""
        result = self.dst_surface.blit(self.src_surface, dest=pygame.Rect((1, 1, 1, 1)))
        self.assertIsInstance(result, pygame.Rect)
        self.assertEqual(result.size, (63, 63))
        self.assertEqual(self.dst_surface.get_at((0, 0)), (0, 0, 0))
        self.assertEqual(self.dst_surface.get_at((63, 0)), (0, 0, 0))
        self.assertEqual(self.dst_surface.get_at((0, 63)), (0, 0, 0))
        self.assertEqual(self.dst_surface.get_at((1, 1)), (255, 255, 255))
        self.assertEqual(self.dst_surface.get_at((63, 63)), (255, 255, 255))

    def test_blit_area_contraint(self):
        """Testing area constraint"""
        result = self.dst_surface.blit(self.src_surface, dest=pygame.Rect((1, 1, 1, 1)), area=pygame.Rect((2, 2, 2, 2)))
        self.assertIsInstance(result, pygame.Rect)
        self.assertEqual(result.size, (2, 2))
        self.assertEqual(self.dst_surface.get_at((0, 0)), (0, 0, 0))
        self.assertEqual(self.dst_surface.get_at((63, 0)), (0, 0, 0))
        self.assertEqual(self.dst_surface.get_at((0, 63)), (0, 0, 0))
        self.assertEqual(self.dst_surface.get_at((63, 63)), (0, 0, 0))
        self.assertEqual(self.dst_surface.get_at((1, 1)), (255, 255, 255))
        self.assertEqual(self.dst_surface.get_at((2, 2)), (255, 255, 255))
        self.assertEqual(self.dst_surface.get_at((3, 3)), (0, 0, 0))

    def test_blit_zero_overlap(self):
        """Testing zero-overlap condition."""
        result = self.dst_surface.blit(self.src_surface, dest=pygame.Rect((-256, -256, 1, 1)), area=pygame.Rect((2, 2, 256, 256)))
        self.assertIsInstance(result, pygame.Rect)
        self.assertEqual(result.size, (0, 0))
        for k in [(x, x) for x in range(64)]:
            self.assertEqual(self.dst_surface.get_at(k), (0, 0, 0))
        self.assertEqual(self.dst_surface.get_at((63, 0)), (0, 0, 0))
        self.assertEqual(self.dst_surface.get_at((0, 63)), (0, 0, 0))

    def test_blit__SRCALPHA_opaque_source(self):
        src = pygame.Surface((256, 256), SRCALPHA, 32)
        dst = src.copy()
        for i, j in test_utils.rect_area_pts(src.get_rect()):
            dst.set_at((i, j), (i, 0, 0, j))
            src.set_at((i, j), (0, i, 0, 255))
        dst.blit(src, (0, 0))
        for pt in test_utils.rect_area_pts(src.get_rect()):
            self.assertEqual(dst.get_at(pt)[1], src.get_at(pt)[1])

    def test_blit__blit_to_self(self):
        """Test that blit operation works on self, alpha value is
        correct, and that no RGB distortion occurs."""
        test_surface = pygame.Surface((128, 128), SRCALPHA, 32)
        area = test_surface.get_rect()
        for pt, test_color in test_utils.gradient(area.width, area.height):
            test_surface.set_at(pt, test_color)
        reference_surface = test_surface.copy()
        test_surface.blit(test_surface, (0, 0))
        for x in range(area.width):
            for y in range(area.height):
                r, g, b, a = reference_color = reference_surface.get_at((x, y))
                expected_color = (r, g, b, a + a * ((256 - a) // 256))
                self.assertEqual(reference_color, expected_color)
        self.assertEqual(reference_surface.get_rect(), test_surface.get_rect())

    def test_blit__SRCALPHA_to_SRCALPHA_non_zero(self):
        """Tests blitting a nonzero alpha surface to another nonzero alpha surface
        both straight alpha compositing method. Test is fuzzy (+/- 1/256) to account for
        different implementations in SDL1 and SDL2.
        """
        size = (32, 32)

        def check_color_diff(color1, color2):
            """Returns True if two colors are within (1, 1, 1, 1) of each other."""
            for val in color1 - color2:
                if abs(val) > 1:
                    return False
            return True

        def high_a_onto_low(high, low):
            """Tests straight alpha case. Source is low alpha, destination is high alpha"""
            high_alpha_surface = pygame.Surface(size, pygame.SRCALPHA, 32)
            low_alpha_surface = high_alpha_surface.copy()
            high_alpha_color = Color((high, high, low, high))
            low_alpha_color = Color((high, low, low, low))
            high_alpha_surface.fill(high_alpha_color)
            low_alpha_surface.fill(low_alpha_color)
            high_alpha_surface.blit(low_alpha_surface, (0, 0))
            expected_color = low_alpha_color + Color(tuple((x * (255 - low_alpha_color.a) // 255 for x in high_alpha_color)))
            self.assertTrue(check_color_diff(high_alpha_surface.get_at((0, 0)), expected_color))

        def low_a_onto_high(high, low):
            """Tests straight alpha case. Source is high alpha, destination is low alpha"""
            high_alpha_surface = pygame.Surface(size, pygame.SRCALPHA, 32)
            low_alpha_surface = high_alpha_surface.copy()
            high_alpha_color = Color((high, high, low, high))
            low_alpha_color = Color((high, low, low, low))
            high_alpha_surface.fill(high_alpha_color)
            low_alpha_surface.fill(low_alpha_color)
            low_alpha_surface.blit(high_alpha_surface, (0, 0))
            expected_color = high_alpha_color + Color(tuple((x * (255 - high_alpha_color.a) // 255 for x in low_alpha_color)))
            self.assertTrue(check_color_diff(low_alpha_surface.get_at((0, 0)), expected_color))
        for low_a in range(0, 128):
            for high_a in range(128, 256):
                high_a_onto_low(high_a, low_a)
                low_a_onto_high(high_a, low_a)

    def test_blit__SRCALPHA32_to_8(self):
        target = pygame.Surface((11, 8), 0, 8)
        test_color = target.get_palette_at(2)
        source = pygame.Surface((1, 1), pygame.SRCALPHA, 32)
        source.set_at((0, 0), test_color)
        target.blit(source, (0, 0))