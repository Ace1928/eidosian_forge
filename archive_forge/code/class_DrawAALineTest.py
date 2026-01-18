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
class DrawAALineTest(AALineMixin, DrawTestCase):
    """Test draw module function aaline.

    This class inherits the general tests from AALineMixin. It is also the
    class to add any draw.aaline specific tests to.
    """

    def test_aaline_endianness(self):
        """test color component order"""
        for depth in (24, 32):
            surface = pygame.Surface((5, 3), 0, depth)
            surface.fill(pygame.Color(0, 0, 0))
            self.draw_aaline(surface, pygame.Color(255, 0, 0), (0, 1), (2, 1), 1)
            self.assertGreater(surface.get_at((1, 1)).r, 0, 'there should be red here')
            surface.fill(pygame.Color(0, 0, 0))
            self.draw_aaline(surface, pygame.Color(0, 0, 255), (0, 1), (2, 1), 1)
            self.assertGreater(surface.get_at((1, 1)).b, 0, 'there should be blue here')

    def _check_antialiasing(self, from_point, to_point, should, check_points, set_endpoints=True):
        """Draw a line between two points and check colors of check_points."""
        if set_endpoints:
            should[from_point] = should[to_point] = FG_GREEN

        def check_one_direction(from_point, to_point, should):
            self.draw_aaline(self.surface, FG_GREEN, from_point, to_point, True)
            for pt in check_points:
                color = should.get(pt, BG_RED)
                with self.subTest(from_pt=from_point, pt=pt, to=to_point):
                    self.assertEqual(self.surface.get_at(pt), color)
            draw.rect(self.surface, BG_RED, (0, 0, 10, 10), 0)
        check_one_direction(from_point, to_point, should)
        if from_point != to_point:
            check_one_direction(to_point, from_point, should)

    def test_short_non_antialiased_lines(self):
        """test very short not anti aliased lines in all directions."""
        self.surface = pygame.Surface((10, 10))
        draw.rect(self.surface, BG_RED, (0, 0, 10, 10), 0)
        check_points = [(i, j) for i in range(3, 8) for j in range(3, 8)]

        def check_both_directions(from_pt, to_pt, other_points):
            should = {pt: FG_GREEN for pt in other_points}
            self._check_antialiasing(from_pt, to_pt, should, check_points)
        check_both_directions((5, 5), (5, 5), [])
        check_both_directions((4, 7), (5, 7), [])
        check_both_directions((5, 4), (7, 4), [(6, 4)])
        check_both_directions((5, 5), (5, 6), [])
        check_both_directions((6, 4), (6, 6), [(6, 5)])
        check_both_directions((5, 5), (6, 6), [])
        check_both_directions((5, 5), (7, 7), [(6, 6)])
        check_both_directions((5, 6), (6, 5), [])
        check_both_directions((6, 4), (4, 6), [(5, 5)])

    def test_short_line_anti_aliasing(self):
        self.surface = pygame.Surface((10, 10))
        draw.rect(self.surface, BG_RED, (0, 0, 10, 10), 0)
        check_points = [(i, j) for i in range(3, 8) for j in range(3, 8)]

        def check_both_directions(from_pt, to_pt, should):
            self._check_antialiasing(from_pt, to_pt, should, check_points)
        brown = (127, 127, 0)
        reddish = (191, 63, 0)
        greenish = (63, 191, 0)
        check_both_directions((4, 4), (6, 5), {(5, 4): brown, (5, 5): brown})
        check_both_directions((4, 5), (6, 4), {(5, 4): brown, (5, 5): brown})
        check_both_directions((4, 4), (5, 6), {(4, 5): brown, (5, 5): brown})
        check_both_directions((5, 4), (4, 6), {(4, 5): brown, (5, 5): brown})
        check_points = [(i, j) for i in range(2, 9) for j in range(2, 9)]
        should = {(4, 3): greenish, (5, 3): brown, (6, 3): reddish, (4, 4): reddish, (5, 4): brown, (6, 4): greenish}
        check_both_directions((3, 3), (7, 4), should)
        should = {(4, 3): reddish, (5, 3): brown, (6, 3): greenish, (4, 4): greenish, (5, 4): brown, (6, 4): reddish}
        check_both_directions((3, 4), (7, 3), should)
        should = {(4, 4): greenish, (4, 5): brown, (4, 6): reddish, (5, 4): reddish, (5, 5): brown, (5, 6): greenish}
        check_both_directions((4, 3), (5, 7), should)
        should = {(4, 4): reddish, (4, 5): brown, (4, 6): greenish, (5, 4): greenish, (5, 5): brown, (5, 6): reddish}
        check_both_directions((5, 3), (4, 7), should)

    def test_anti_aliasing_float_coordinates(self):
        """Float coordinates should be blended smoothly."""
        self.surface = pygame.Surface((10, 10))
        draw.rect(self.surface, BG_RED, (0, 0, 10, 10), 0)
        check_points = [(i, j) for i in range(5) for j in range(5)]
        brown = (127, 127, 0)
        reddish = (191, 63, 0)
        greenish = (63, 191, 0)
        expected = {(2, 2): FG_GREEN}
        self._check_antialiasing((1.5, 2), (1.5, 2), expected, check_points, set_endpoints=False)
        expected = {(2, 3): FG_GREEN}
        self._check_antialiasing((2.49, 2.7), (2.49, 2.7), expected, check_points, set_endpoints=False)
        expected = {(1, 2): brown, (2, 2): FG_GREEN}
        self._check_antialiasing((1.5, 2), (2, 2), expected, check_points, set_endpoints=False)
        expected = {(1, 2): brown, (2, 2): FG_GREEN, (3, 2): brown}
        self._check_antialiasing((1.5, 2), (2.5, 2), expected, check_points, set_endpoints=False)
        expected = {(2, 2): brown, (1, 2): FG_GREEN}
        self._check_antialiasing((1, 2), (1.5, 2), expected, check_points, set_endpoints=False)
        expected = {(1, 2): brown, (2, 2): greenish}
        self._check_antialiasing((1.5, 2), (1.75, 2), expected, check_points, set_endpoints=False)
        expected = {(x, y): brown for x in range(2, 5) for y in (1, 2)}
        self._check_antialiasing((2, 1.5), (4, 1.5), expected, check_points, set_endpoints=False)
        expected = {(2, 1): brown, (2, 2): FG_GREEN, (2, 3): brown}
        self._check_antialiasing((2, 1.5), (2, 2.5), expected, check_points, set_endpoints=False)
        expected = {(2, 1): brown, (2, 2): greenish}
        self._check_antialiasing((2, 1.5), (2, 1.75), expected, check_points, set_endpoints=False)
        expected = {(x, y): brown for x in (1, 2) for y in range(2, 5)}
        self._check_antialiasing((1.5, 2), (1.5, 4), expected, check_points, set_endpoints=False)
        expected = {(1, 1): brown, (2, 2): FG_GREEN, (3, 3): brown}
        self._check_antialiasing((1.5, 1.5), (2.5, 2.5), expected, check_points, set_endpoints=False)
        expected = {(3, 1): brown, (2, 2): FG_GREEN, (1, 3): brown}
        self._check_antialiasing((2.5, 1.5), (1.5, 2.5), expected, check_points, set_endpoints=False)
        expected = {(2, 1): brown, (2, 2): brown, (3, 2): brown, (3, 3): brown}
        self._check_antialiasing((2, 1.5), (3, 2.5), expected, check_points, set_endpoints=False)
        expected = {(2, 1): greenish, (2, 2): reddish, (3, 2): greenish, (3, 3): reddish, (4, 3): greenish, (4, 4): reddish}
        self._check_antialiasing((2, 1.25), (4, 3.25), expected, check_points, set_endpoints=False)

    def test_anti_aliasing_at_and_outside_the_border(self):
        """Ensures antialiasing works correct at a surface's borders."""
        self.surface = pygame.Surface((10, 10))
        draw.rect(self.surface, BG_RED, (0, 0, 10, 10), 0)
        check_points = [(i, j) for i in range(10) for j in range(10)]
        reddish = (191, 63, 0)
        brown = (127, 127, 0)
        greenish = (63, 191, 0)
        from_point, to_point = ((3, 3), (7, 4))
        should = {(4, 3): greenish, (5, 3): brown, (6, 3): reddish, (4, 4): reddish, (5, 4): brown, (6, 4): greenish}
        for dx, dy in ((-4, 0), (4, 0), (0, -5), (0, -4), (0, -3), (0, 5), (0, 6), (0, 7), (-4, -4), (-4, -3), (-3, -4)):
            first = (from_point[0] + dx, from_point[1] + dy)
            second = (to_point[0] + dx, to_point[1] + dy)
            expected = {(x + dx, y + dy): color for (x, y), color in should.items()}
            self._check_antialiasing(first, second, expected, check_points)