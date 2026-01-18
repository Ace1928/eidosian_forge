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
class DrawModuleTest(unittest.TestCase):
    """General draw module tests."""

    def test_path_data_validation(self):
        """Test validation of multi-point drawing methods.

        See bug #521
        """
        surf = pygame.Surface((5, 5))
        rect = pygame.Rect(0, 0, 5, 5)
        bad_values = ('text', b'bytes', 1 + 1j, object(), lambda x: x)
        bad_points = list(bad_values) + [(1,), (1, 2, 3)]
        bad_points.extend(((1, v) for v in bad_values))
        good_path = [(1, 1), (1, 3), (3, 3), (3, 1)]
        check_pts = [(x, y) for x in range(5) for y in range(5)]
        for method, is_polgon in ((draw.lines, 0), (draw.aalines, 0), (draw.polygon, 1)):
            for val in bad_values:
                draw.rect(surf, RED, rect, 0)
                with self.assertRaises(TypeError):
                    if is_polgon:
                        method(surf, GREEN, [val] + good_path, 0)
                    else:
                        method(surf, GREEN, True, [val] + good_path)
                self.assertTrue(all((surf.get_at(pt) == RED for pt in check_pts)))
                draw.rect(surf, RED, rect, 0)
                with self.assertRaises(TypeError):
                    path = good_path[:2] + [val] + good_path[2:]
                    if is_polgon:
                        method(surf, GREEN, path, 0)
                    else:
                        method(surf, GREEN, True, path)
                self.assertTrue(all((surf.get_at(pt) == RED for pt in check_pts)))

    def test_color_validation(self):
        surf = pygame.Surface((10, 10))
        colors = (123456, (1, 10, 100), RED, '#ab12df', 'red')
        points = ((0, 0), (1, 1), (1, 0))
        for col in colors:
            draw.line(surf, col, (0, 0), (1, 1))
            draw.aaline(surf, col, (0, 0), (1, 1))
            draw.aalines(surf, col, True, points)
            draw.lines(surf, col, True, points)
            draw.arc(surf, col, pygame.Rect(0, 0, 3, 3), 15, 150)
            draw.ellipse(surf, col, pygame.Rect(0, 0, 3, 6), 1)
            draw.circle(surf, col, (7, 3), 2)
            draw.polygon(surf, col, points, 0)
        for col in (1.256, object(), None):
            with self.assertRaises(TypeError):
                draw.line(surf, col, (0, 0), (1, 1))
            with self.assertRaises(TypeError):
                draw.aaline(surf, col, (0, 0), (1, 1))
            with self.assertRaises(TypeError):
                draw.aalines(surf, col, True, points)
            with self.assertRaises(TypeError):
                draw.lines(surf, col, True, points)
            with self.assertRaises(TypeError):
                draw.arc(surf, col, pygame.Rect(0, 0, 3, 3), 15, 150)
            with self.assertRaises(TypeError):
                draw.ellipse(surf, col, pygame.Rect(0, 0, 3, 6), 1)
            with self.assertRaises(TypeError):
                draw.circle(surf, col, (7, 3), 2)
            with self.assertRaises(TypeError):
                draw.polygon(surf, col, points, 0)