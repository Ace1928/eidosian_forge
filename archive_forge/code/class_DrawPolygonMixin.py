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
class DrawPolygonMixin:
    """Mixin tests for drawing polygons.

    This class contains all the general polygon drawing tests.
    """

    def setUp(self):
        self.surface = pygame.Surface((20, 20))

    def test_polygon__args(self):
        """Ensures draw polygon accepts the correct args."""
        bounds_rect = self.draw_polygon(pygame.Surface((3, 3)), (0, 10, 0, 50), ((0, 0), (1, 1), (2, 2)), 1)
        self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_polygon__args_without_width(self):
        """Ensures draw polygon accepts the args without a width."""
        bounds_rect = self.draw_polygon(pygame.Surface((2, 2)), (0, 0, 0, 50), ((0, 0), (1, 1), (2, 2)))
        self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_polygon__kwargs(self):
        """Ensures draw polygon accepts the correct kwargs
        with and without a width arg.
        """
        surface = pygame.Surface((4, 4))
        color = pygame.Color('yellow')
        points = ((0, 0), (1, 1), (2, 2))
        kwargs_list = [{'surface': surface, 'color': color, 'points': points, 'width': 1}, {'surface': surface, 'color': color, 'points': points}]
        for kwargs in kwargs_list:
            bounds_rect = self.draw_polygon(**kwargs)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_polygon__kwargs_order_independent(self):
        """Ensures draw polygon's kwargs are not order dependent."""
        bounds_rect = self.draw_polygon(color=(10, 20, 30), surface=pygame.Surface((3, 2)), width=0, points=((0, 1), (1, 2), (2, 3)))
        self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_polygon__args_missing(self):
        """Ensures draw polygon detects any missing required args."""
        surface = pygame.Surface((1, 1))
        color = pygame.Color('blue')
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_polygon(surface, color)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_polygon(surface)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_polygon()

    def test_polygon__kwargs_missing(self):
        """Ensures draw polygon detects any missing required kwargs."""
        kwargs = {'surface': pygame.Surface((1, 2)), 'color': pygame.Color('red'), 'points': ((2, 1), (2, 2), (2, 3)), 'width': 1}
        for name in ('points', 'color', 'surface'):
            invalid_kwargs = dict(kwargs)
            invalid_kwargs.pop(name)
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_polygon(**invalid_kwargs)

    def test_polygon__arg_invalid_types(self):
        """Ensures draw polygon detects invalid arg types."""
        surface = pygame.Surface((2, 2))
        color = pygame.Color('blue')
        points = ((0, 1), (1, 2), (1, 3))
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_polygon(surface, color, points, '1')
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_polygon(surface, color, (1, 2, 3))
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_polygon(surface, 2.3, points)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_polygon((1, 2, 3, 4), color, points)

    def test_polygon__kwarg_invalid_types(self):
        """Ensures draw polygon detects invalid kwarg types."""
        surface = pygame.Surface((3, 3))
        color = pygame.Color('green')
        points = ((0, 0), (1, 0), (2, 0))
        width = 1
        kwargs_list = [{'surface': pygame.Surface, 'color': color, 'points': points, 'width': width}, {'surface': surface, 'color': 2.3, 'points': points, 'width': width}, {'surface': surface, 'color': color, 'points': ((1,), (1,), (1,)), 'width': width}, {'surface': surface, 'color': color, 'points': points, 'width': 1.2}]
        for kwargs in kwargs_list:
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_polygon(**kwargs)

    def test_polygon__kwarg_invalid_name(self):
        """Ensures draw polygon detects invalid kwarg names."""
        surface = pygame.Surface((2, 3))
        color = pygame.Color('cyan')
        points = ((1, 1), (1, 2), (1, 3))
        kwargs_list = [{'surface': surface, 'color': color, 'points': points, 'width': 1, 'invalid': 1}, {'surface': surface, 'color': color, 'points': points, 'invalid': 1}]
        for kwargs in kwargs_list:
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_polygon(**kwargs)

    def test_polygon__args_and_kwargs(self):
        """Ensures draw polygon accepts a combination of args/kwargs"""
        surface = pygame.Surface((3, 1))
        color = (255, 255, 0, 0)
        points = ((0, 1), (1, 2), (2, 3))
        width = 0
        kwargs = {'surface': surface, 'color': color, 'points': points, 'width': width}
        for name in ('surface', 'color', 'points', 'width'):
            kwargs.pop(name)
            if 'surface' == name:
                bounds_rect = self.draw_polygon(surface, **kwargs)
            elif 'color' == name:
                bounds_rect = self.draw_polygon(surface, color, **kwargs)
            elif 'points' == name:
                bounds_rect = self.draw_polygon(surface, color, points, **kwargs)
            else:
                bounds_rect = self.draw_polygon(surface, color, points, width, **kwargs)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_polygon__valid_width_values(self):
        """Ensures draw polygon accepts different width values."""
        surface_color = pygame.Color('white')
        surface = pygame.Surface((3, 4))
        color = (10, 20, 30, 255)
        kwargs = {'surface': surface, 'color': color, 'points': ((1, 1), (2, 1), (2, 2), (1, 2)), 'width': None}
        pos = kwargs['points'][0]
        for width in (-100, -10, -1, 0, 1, 10, 100):
            surface.fill(surface_color)
            kwargs['width'] = width
            expected_color = color if width >= 0 else surface_color
            bounds_rect = self.draw_polygon(**kwargs)
            self.assertEqual(surface.get_at(pos), expected_color)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_polygon__valid_points_format(self):
        """Ensures draw polygon accepts different points formats."""
        expected_color = (10, 20, 30, 255)
        surface_color = pygame.Color('white')
        surface = pygame.Surface((3, 4))
        kwargs = {'surface': surface, 'color': expected_color, 'points': None, 'width': 0}
        point_types = ((tuple, tuple, tuple, tuple), (list, list, list, list), (Vector2, Vector2, Vector2, Vector2), (list, Vector2, tuple, Vector2))
        point_values = (((1, 1), (2, 1), (2, 2), (1, 2)), ((1, 1), (2.2, 1), (2.1, 2.2), (1, 2.1)))
        seq_types = (tuple, list)
        for point_type in point_types:
            for values in point_values:
                check_pos = values[0]
                points = [point_type[i](pt) for i, pt in enumerate(values)]
                for seq_type in seq_types:
                    surface.fill(surface_color)
                    kwargs['points'] = seq_type(points)
                    bounds_rect = self.draw_polygon(**kwargs)
                    self.assertEqual(surface.get_at(check_pos), expected_color)
                    self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_polygon__invalid_points_formats(self):
        """Ensures draw polygon handles invalid points formats correctly."""
        kwargs = {'surface': pygame.Surface((4, 4)), 'color': pygame.Color('red'), 'points': None, 'width': 0}
        points_fmts = (((1, 1), (2, 1), (2,)), ((1, 1), (2, 1), (2, 2, 2)), ((1, 1), (2, 1), (2, '2')), ((1, 1), (2, 1), {2, 3}), ((1, 1), (2, 1), dict(((2, 2), (3, 3)))), {(1, 1), (2, 1), (2, 2), (1, 2)}, dict(((1, 1), (2, 2), (3, 3), (4, 4))))
        for points in points_fmts:
            kwargs['points'] = points
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_polygon(**kwargs)

    def test_polygon__invalid_points_values(self):
        """Ensures draw polygon handles invalid points values correctly."""
        kwargs = {'surface': pygame.Surface((4, 4)), 'color': pygame.Color('red'), 'points': None, 'width': 0}
        points_fmts = (tuple(), ((1, 1),), ((1, 1), (2, 1)))
        for points in points_fmts:
            for seq_type in (tuple, list):
                kwargs['points'] = seq_type(points)
                with self.assertRaises(ValueError):
                    bounds_rect = self.draw_polygon(**kwargs)

    def test_polygon__valid_color_formats(self):
        """Ensures draw polygon accepts different color formats."""
        green_color = pygame.Color('green')
        surface_color = pygame.Color('black')
        surface = pygame.Surface((3, 4))
        kwargs = {'surface': surface, 'color': None, 'points': ((1, 1), (2, 1), (2, 2), (1, 2)), 'width': 0}
        pos = kwargs['points'][0]
        greens = ((0, 255, 0), (0, 255, 0, 255), surface.map_rgb(green_color), green_color)
        for color in greens:
            surface.fill(surface_color)
            kwargs['color'] = color
            if isinstance(color, int):
                expected_color = surface.unmap_rgb(color)
            else:
                expected_color = green_color
            bounds_rect = self.draw_polygon(**kwargs)
            self.assertEqual(surface.get_at(pos), expected_color)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_polygon__invalid_color_formats(self):
        """Ensures draw polygon handles invalid color formats correctly."""
        kwargs = {'surface': pygame.Surface((4, 3)), 'color': None, 'points': ((1, 1), (2, 1), (2, 2), (1, 2)), 'width': 0}
        for expected_color in (2.3, self):
            kwargs['color'] = expected_color
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_polygon(**kwargs)

    def test_draw_square(self):
        self.draw_polygon(self.surface, RED, SQUARE, 0)
        for x in range(4):
            for y in range(4):
                self.assertEqual(self.surface.get_at((x, y)), RED)

    def test_draw_diamond(self):
        pygame.draw.rect(self.surface, RED, (0, 0, 10, 10), 0)
        self.draw_polygon(self.surface, GREEN, DIAMOND, 0)
        for x, y in DIAMOND:
            self.assertEqual(self.surface.get_at((x, y)), GREEN, msg=str((x, y)))
        for x in range(2, 5):
            for y in range(2, 5):
                self.assertEqual(self.surface.get_at((x, y)), GREEN)

    def test_1_pixel_high_or_wide_shapes(self):
        pygame.draw.rect(self.surface, RED, (0, 0, 10, 10), 0)
        self.draw_polygon(self.surface, GREEN, [(x, 2) for x, _y in CROSS], 0)
        cross_size = 6
        for x in range(cross_size + 1):
            self.assertEqual(self.surface.get_at((x, 1)), RED)
            self.assertEqual(self.surface.get_at((x, 2)), GREEN)
            self.assertEqual(self.surface.get_at((x, 3)), RED)
        pygame.draw.rect(self.surface, RED, (0, 0, 10, 10), 0)
        self.draw_polygon(self.surface, GREEN, [(x, 5) for x, _y in CROSS], 1)
        for x in range(cross_size + 1):
            self.assertEqual(self.surface.get_at((x, 4)), RED)
            self.assertEqual(self.surface.get_at((x, 5)), GREEN)
            self.assertEqual(self.surface.get_at((x, 6)), RED)
        pygame.draw.rect(self.surface, RED, (0, 0, 10, 10), 0)
        self.draw_polygon(self.surface, GREEN, [(3, y) for _x, y in CROSS], 0)
        for y in range(cross_size + 1):
            self.assertEqual(self.surface.get_at((2, y)), RED)
            self.assertEqual(self.surface.get_at((3, y)), GREEN)
            self.assertEqual(self.surface.get_at((4, y)), RED)
        pygame.draw.rect(self.surface, RED, (0, 0, 10, 10), 0)
        self.draw_polygon(self.surface, GREEN, [(4, y) for _x, y in CROSS], 1)
        for y in range(cross_size + 1):
            self.assertEqual(self.surface.get_at((3, y)), RED)
            self.assertEqual(self.surface.get_at((4, y)), GREEN)
            self.assertEqual(self.surface.get_at((5, y)), RED)

    def test_draw_symetric_cross(self):
        """non-regression on issue #234 : x and y where handled inconsistently.

        Also, the result is/was different whether we fill or not the polygon.
        """
        pygame.draw.rect(self.surface, RED, (0, 0, 10, 10), 0)
        self.draw_polygon(self.surface, GREEN, CROSS, 1)
        inside = [(x, 3) for x in range(1, 6)] + [(3, y) for y in range(1, 6)]
        for x in range(10):
            for y in range(10):
                if (x, y) in inside:
                    self.assertEqual(self.surface.get_at((x, y)), RED)
                elif x in range(2, 5) and y < 7 or (y in range(2, 5) and x < 7):
                    self.assertEqual(self.surface.get_at((x, y)), GREEN)
                else:
                    self.assertEqual(self.surface.get_at((x, y)), RED)
        pygame.draw.rect(self.surface, RED, (0, 0, 10, 10), 0)
        self.draw_polygon(self.surface, GREEN, CROSS, 0)
        inside = [(x, 3) for x in range(1, 6)] + [(3, y) for y in range(1, 6)]
        for x in range(10):
            for y in range(10):
                if x in range(2, 5) and y < 7 or (y in range(2, 5) and x < 7):
                    self.assertEqual(self.surface.get_at((x, y)), GREEN, msg=str((x, y)))
                else:
                    self.assertEqual(self.surface.get_at((x, y)), RED)

    def test_illumine_shape(self):
        """non-regression on issue #313"""
        rect = pygame.Rect((0, 0, 20, 20))
        path_data = [(0, 0), (rect.width - 1, 0), (rect.width - 5, 5 - 1), (5 - 1, 5 - 1), (5 - 1, rect.height - 5), (0, rect.height - 1)]
        pygame.draw.rect(self.surface, RED, (0, 0, 20, 20), 0)
        self.draw_polygon(self.surface, GREEN, path_data[:4], 0)
        for x in range(20):
            self.assertEqual(self.surface.get_at((x, 0)), GREEN)
        for x in range(4, rect.width - 5 + 1):
            self.assertEqual(self.surface.get_at((x, 4)), GREEN)
        pygame.draw.rect(self.surface, RED, (0, 0, 20, 20), 0)
        self.draw_polygon(self.surface, GREEN, path_data, 0)
        for x in range(4, rect.width - 5 + 1):
            self.assertEqual(self.surface.get_at((x, 4)), GREEN)

    def test_invalid_points(self):
        self.assertRaises(TypeError, lambda: self.draw_polygon(self.surface, RED, ((0, 0), (0, 20), (20, 20), 20), 0))

    def test_polygon__bounding_rect(self):
        """Ensures draw polygon returns the correct bounding rect.

        Tests polygons on and off the surface and a range of width/thickness
        values.
        """
        polygon_color = pygame.Color('red')
        surf_color = pygame.Color('black')
        min_width = min_height = 5
        max_width = max_height = 7
        sizes = ((min_width, min_height), (max_width, max_height))
        surface = pygame.Surface((20, 20), 0, 32)
        surf_rect = surface.get_rect()
        big_rect = surf_rect.inflate(min_width * 2 + 1, min_height * 2 + 1)
        for pos in rect_corners_mids_and_center(surf_rect) + rect_corners_mids_and_center(big_rect):
            for attr in RECT_POSITION_ATTRIBUTES:
                for width, height in sizes:
                    pos_rect = pygame.Rect((0, 0), (width, height))
                    setattr(pos_rect, attr, pos)
                    vertices = (pos_rect.midleft, pos_rect.midtop, pos_rect.bottomright)
                    for thickness in range(4):
                        surface.fill(surf_color)
                        bounding_rect = self.draw_polygon(surface, polygon_color, vertices, thickness)
                        expected_rect = create_bounding_rect(surface, surf_color, vertices[0])
                        self.assertEqual(bounding_rect, expected_rect, f'thickness={thickness}')

    def test_polygon__surface_clip(self):
        """Ensures draw polygon respects a surface's clip area.

        Tests drawing the polygon filled and unfilled.
        """
        surfw = surfh = 30
        polygon_color = pygame.Color('red')
        surface_color = pygame.Color('green')
        surface = pygame.Surface((surfw, surfh))
        surface.fill(surface_color)
        clip_rect = pygame.Rect((0, 0), (8, 10))
        clip_rect.center = surface.get_rect().center
        pos_rect = clip_rect.copy()
        for width in (0, 1):
            for center in rect_corners_mids_and_center(clip_rect):
                pos_rect.center = center
                vertices = (pos_rect.topleft, pos_rect.topright, pos_rect.bottomright, pos_rect.bottomleft)
                surface.set_clip(None)
                surface.fill(surface_color)
                self.draw_polygon(surface, polygon_color, vertices, width)
                expected_pts = get_color_points(surface, polygon_color, clip_rect)
                surface.fill(surface_color)
                surface.set_clip(clip_rect)
                self.draw_polygon(surface, polygon_color, vertices, width)
                surface.lock()
                for pt in ((x, y) for x in range(surfw) for y in range(surfh)):
                    if pt in expected_pts:
                        expected_color = polygon_color
                    else:
                        expected_color = surface_color
                    self.assertEqual(surface.get_at(pt), expected_color, pt)
                surface.unlock()