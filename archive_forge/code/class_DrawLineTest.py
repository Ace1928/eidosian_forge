import math
import unittest
import sys
import warnings
import pygame
from pygame import draw
from pygame import draw_py
from pygame.locals import SRCALPHA
from pygame.tests import test_utils
from pygame.math import Vector2
class DrawLineTest(LineMixin, DrawTestCase):
    """Test draw module function line.

    This class inherits the general tests from LineMixin. It is also the class
    to add any draw.line specific tests to.
    """

    def test_line_endianness(self):
        """test color component order"""
        for depth in (24, 32):
            surface = pygame.Surface((5, 3), 0, depth)
            surface.fill(pygame.Color(0, 0, 0))
            self.draw_line(surface, pygame.Color(255, 0, 0), (0, 1), (2, 1), 1)
            self.assertGreater(surface.get_at((1, 1)).r, 0, 'there should be red here')
            surface.fill(pygame.Color(0, 0, 0))
            self.draw_line(surface, pygame.Color(0, 0, 255), (0, 1), (2, 1), 1)
            self.assertGreater(surface.get_at((1, 1)).b, 0, 'there should be blue here')

    def test_line(self):
        self.surf_size = (320, 200)
        self.surf = pygame.Surface(self.surf_size, pygame.SRCALPHA)
        self.color = (1, 13, 24, 205)
        drawn = draw.line(self.surf, self.color, (1, 0), (200, 0))
        self.assertEqual(drawn.right, 201, 'end point arg should be (or at least was) inclusive')
        for pt in test_utils.rect_area_pts(drawn):
            self.assertEqual(self.surf.get_at(pt), self.color)
        for pt in test_utils.rect_outer_bounds(drawn):
            self.assertNotEqual(self.surf.get_at(pt), self.color)
        line_width = 2
        offset = 5
        a = (offset, offset)
        b = (self.surf_size[0] - offset, a[1])
        c = (a[0], self.surf_size[1] - offset)
        d = (b[0], c[1])
        e = (a[0] + offset, c[1])
        f = (b[0], c[0] + 5)
        lines = [(a, d), (b, c), (c, b), (d, a), (a, b), (b, a), (a, c), (c, a), (a, e), (e, a), (a, f), (f, a), (a, a)]
        for p1, p2 in lines:
            msg = f'{p1} - {p2}'
            if p1[0] <= p2[0]:
                plow = p1
                phigh = p2
            else:
                plow = p2
                phigh = p1
            self.surf.fill((0, 0, 0))
            rec = draw.line(self.surf, (255, 255, 255), p1, p2, line_width)
            xinc = yinc = 0
            if abs(p1[0] - p2[0]) > abs(p1[1] - p2[1]):
                yinc = 1
            else:
                xinc = 1
            for i in range(line_width):
                p = (p1[0] + xinc * i, p1[1] + yinc * i)
                self.assertEqual(self.surf.get_at(p), (255, 255, 255), msg)
                p = (p2[0] + xinc * i, p2[1] + yinc * i)
                self.assertEqual(self.surf.get_at(p), (255, 255, 255), msg)
            p = (plow[0] - 1, plow[1])
            self.assertEqual(self.surf.get_at(p), (0, 0, 0), msg)
            p = (plow[0] + xinc * line_width, plow[1] + yinc * line_width)
            self.assertEqual(self.surf.get_at(p), (0, 0, 0), msg)
            p = (phigh[0] + xinc * line_width, phigh[1] + yinc * line_width)
            self.assertEqual(self.surf.get_at(p), (0, 0, 0), msg)
            if p1[0] < p2[0]:
                rx = p1[0]
            else:
                rx = p2[0]
            if p1[1] < p2[1]:
                ry = p1[1]
            else:
                ry = p2[1]
            w = abs(p2[0] - p1[0]) + 1 + xinc * (line_width - 1)
            h = abs(p2[1] - p1[1]) + 1 + yinc * (line_width - 1)
            msg += f', {rec}'
            self.assertEqual(rec, (rx, ry, w, h), msg)

    def test_line_for_gaps(self):
        width = 200
        height = 200
        surf = pygame.Surface((width, height), pygame.SRCALPHA)

        def white_surrounded_pixels(x, y):
            offsets = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            WHITE = (255, 255, 255, 255)
            return len([1 for dx, dy in offsets if surf.get_at((x + dx, y + dy)) == WHITE])

        def check_white_line(start, end):
            surf.fill((0, 0, 0))
            pygame.draw.line(surf, (255, 255, 255), start, end, 30)
            BLACK = (0, 0, 0, 255)
            for x in range(1, width - 1):
                for y in range(1, height - 1):
                    if surf.get_at((x, y)) == BLACK:
                        self.assertTrue(white_surrounded_pixels(x, y) < 3)
        check_white_line((50, 50), (140, 0))
        check_white_line((50, 50), (0, 120))
        check_white_line((50, 50), (199, 198))