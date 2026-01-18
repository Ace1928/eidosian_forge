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
class DrawRectMixin:
    """Mixin tests for drawing rects.

    This class contains all the general rect drawing tests.
    """

    def test_rect__args(self):
        """Ensures draw rect accepts the correct args."""
        bounds_rect = self.draw_rect(pygame.Surface((2, 2)), (20, 10, 20, 150), pygame.Rect((0, 0), (1, 1)), 2, 1, 2, 3, 4, 5)
        self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_rect__args_without_width(self):
        """Ensures draw rect accepts the args without a width and borders."""
        bounds_rect = self.draw_rect(pygame.Surface((3, 5)), (0, 0, 0, 255), pygame.Rect((0, 0), (1, 1)))
        self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_rect__kwargs(self):
        """Ensures draw rect accepts the correct kwargs
        with and without a width and border_radius arg.
        """
        kwargs_list = [{'surface': pygame.Surface((5, 5)), 'color': pygame.Color('red'), 'rect': pygame.Rect((0, 0), (1, 2)), 'width': 1, 'border_radius': 10, 'border_top_left_radius': 5, 'border_top_right_radius': 20, 'border_bottom_left_radius': 15, 'border_bottom_right_radius': 0}, {'surface': pygame.Surface((1, 2)), 'color': (0, 100, 200), 'rect': (0, 0, 1, 1)}]
        for kwargs in kwargs_list:
            bounds_rect = self.draw_rect(**kwargs)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_rect__kwargs_order_independent(self):
        """Ensures draw rect's kwargs are not order dependent."""
        bounds_rect = self.draw_rect(color=(0, 1, 2), border_radius=10, surface=pygame.Surface((2, 3)), border_top_left_radius=5, width=-2, border_top_right_radius=20, border_bottom_right_radius=0, rect=pygame.Rect((0, 0), (0, 0)), border_bottom_left_radius=15)
        self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_rect__args_missing(self):
        """Ensures draw rect detects any missing required args."""
        surface = pygame.Surface((1, 1))
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_rect(surface, pygame.Color('white'))
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_rect(surface)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_rect()

    def test_rect__kwargs_missing(self):
        """Ensures draw rect detects any missing required kwargs."""
        kwargs = {'surface': pygame.Surface((1, 3)), 'color': pygame.Color('red'), 'rect': pygame.Rect((0, 0), (2, 2)), 'width': 5, 'border_radius': 10, 'border_top_left_radius': 5, 'border_top_right_radius': 20, 'border_bottom_left_radius': 15, 'border_bottom_right_radius': 0}
        for name in ('rect', 'color', 'surface'):
            invalid_kwargs = dict(kwargs)
            invalid_kwargs.pop(name)
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_rect(**invalid_kwargs)

    def test_rect__arg_invalid_types(self):
        """Ensures draw rect detects invalid arg types."""
        surface = pygame.Surface((3, 3))
        color = pygame.Color('white')
        rect = pygame.Rect((1, 1), (1, 1))
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_rect(surface, color, rect, 2, border_bottom_right_radius='rad')
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_rect(surface, color, rect, 2, border_bottom_left_radius='rad')
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_rect(surface, color, rect, 2, border_top_right_radius='rad')
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_rect(surface, color, rect, 2, border_top_left_radius='draw')
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_rect(surface, color, rect, 2, 'rad')
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_rect(surface, color, rect, '2', 4)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_rect(surface, color, (1, 2, 3), 2, 6)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_rect(surface, 2.3, rect, 3, 8)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_rect(rect, color, rect, 4, 10)

    def test_rect__kwarg_invalid_types(self):
        """Ensures draw rect detects invalid kwarg types."""
        surface = pygame.Surface((2, 3))
        color = pygame.Color('red')
        rect = pygame.Rect((0, 0), (1, 1))
        kwargs_list = [{'surface': pygame.Surface, 'color': color, 'rect': rect, 'width': 1, 'border_radius': 10, 'border_top_left_radius': 5, 'border_top_right_radius': 20, 'border_bottom_left_radius': 15, 'border_bottom_right_radius': 0}, {'surface': surface, 'color': 2.3, 'rect': rect, 'width': 1, 'border_radius': 10, 'border_top_left_radius': 5, 'border_top_right_radius': 20, 'border_bottom_left_radius': 15, 'border_bottom_right_radius': 0}, {'surface': surface, 'color': color, 'rect': (1, 1, 2), 'width': 1, 'border_radius': 10, 'border_top_left_radius': 5, 'border_top_right_radius': 20, 'border_bottom_left_radius': 15, 'border_bottom_right_radius': 0}, {'surface': surface, 'color': color, 'rect': rect, 'width': 1.1, 'border_radius': 10, 'border_top_left_radius': 5, 'border_top_right_radius': 20, 'border_bottom_left_radius': 15, 'border_bottom_right_radius': 0}, {'surface': surface, 'color': color, 'rect': rect, 'width': 1, 'border_radius': 10.5, 'border_top_left_radius': 5, 'border_top_right_radius': 20, 'border_bottom_left_radius': 15, 'border_bottom_right_radius': 0}, {'surface': surface, 'color': color, 'rect': rect, 'width': 1, 'border_radius': 10, 'border_top_left_radius': 5.5, 'border_top_right_radius': 20, 'border_bottom_left_radius': 15, 'border_bottom_right_radius': 0}, {'surface': surface, 'color': color, 'rect': rect, 'width': 1, 'border_radius': 10, 'border_top_left_radius': 5, 'border_top_right_radius': 'a', 'border_bottom_left_radius': 15, 'border_bottom_right_radius': 0}, {'surface': surface, 'color': color, 'rect': rect, 'width': 1, 'border_radius': 10, 'border_top_left_radius': 5, 'border_top_right_radius': 20, 'border_bottom_left_radius': 'c', 'border_bottom_right_radius': 0}, {'surface': surface, 'color': color, 'rect': rect, 'width': 1, 'border_radius': 10, 'border_top_left_radius': 5, 'border_top_right_radius': 20, 'border_bottom_left_radius': 15, 'border_bottom_right_radius': 'd'}]
        for kwargs in kwargs_list:
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_rect(**kwargs)

    def test_rect__kwarg_invalid_name(self):
        """Ensures draw rect detects invalid kwarg names."""
        surface = pygame.Surface((2, 1))
        color = pygame.Color('green')
        rect = pygame.Rect((0, 0), (3, 3))
        kwargs_list = [{'surface': surface, 'color': color, 'rect': rect, 'width': 1, 'border_radius': 10, 'border_top_left_radius': 5, 'border_top_right_radius': 20, 'border_bottom_left_radius': 15, 'border_bottom_right_radius': 0, 'invalid': 1}, {'surface': surface, 'color': color, 'rect': rect, 'invalid': 1}]
        for kwargs in kwargs_list:
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_rect(**kwargs)

    def test_rect__args_and_kwargs(self):
        """Ensures draw rect accepts a combination of args/kwargs"""
        surface = pygame.Surface((3, 1))
        color = (255, 255, 255, 0)
        rect = pygame.Rect((1, 0), (2, 5))
        width = 0
        kwargs = {'surface': surface, 'color': color, 'rect': rect, 'width': width}
        for name in ('surface', 'color', 'rect', 'width'):
            kwargs.pop(name)
            if 'surface' == name:
                bounds_rect = self.draw_rect(surface, **kwargs)
            elif 'color' == name:
                bounds_rect = self.draw_rect(surface, color, **kwargs)
            elif 'rect' == name:
                bounds_rect = self.draw_rect(surface, color, rect, **kwargs)
            else:
                bounds_rect = self.draw_rect(surface, color, rect, width, **kwargs)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_rect__valid_width_values(self):
        """Ensures draw rect accepts different width values."""
        pos = (1, 1)
        surface_color = pygame.Color('black')
        surface = pygame.Surface((3, 4))
        color = (1, 2, 3, 255)
        kwargs = {'surface': surface, 'color': color, 'rect': pygame.Rect(pos, (2, 2)), 'width': None}
        for width in (-1000, -10, -1, 0, 1, 10, 1000):
            surface.fill(surface_color)
            kwargs['width'] = width
            expected_color = color if width >= 0 else surface_color
            bounds_rect = self.draw_rect(**kwargs)
            self.assertEqual(surface.get_at(pos), expected_color)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_rect__valid_rect_formats(self):
        """Ensures draw rect accepts different rect formats."""
        pos = (1, 1)
        expected_color = pygame.Color('yellow')
        surface_color = pygame.Color('black')
        surface = pygame.Surface((3, 4))
        kwargs = {'surface': surface, 'color': expected_color, 'rect': None, 'width': 0}
        rects = (pygame.Rect(pos, (1, 1)), (pos, (2, 2)), (pos[0], pos[1], 3, 3), [pos, (2.1, 2.2)])
        for rect in rects:
            surface.fill(surface_color)
            kwargs['rect'] = rect
            bounds_rect = self.draw_rect(**kwargs)
            self.assertEqual(surface.get_at(pos), expected_color)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_rect__invalid_rect_formats(self):
        """Ensures draw rect handles invalid rect formats correctly."""
        kwargs = {'surface': pygame.Surface((4, 4)), 'color': pygame.Color('red'), 'rect': None, 'width': 0}
        invalid_fmts = ([], [1], [1, 2], [1, 2, 3], [1, 2, 3, 4, 5], {1, 2, 3, 4}, [1, 2, 3, '4'])
        for rect in invalid_fmts:
            kwargs['rect'] = rect
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_rect(**kwargs)

    def test_rect__valid_color_formats(self):
        """Ensures draw rect accepts different color formats."""
        pos = (1, 1)
        red_color = pygame.Color('red')
        surface_color = pygame.Color('black')
        surface = pygame.Surface((3, 4))
        kwargs = {'surface': surface, 'color': None, 'rect': pygame.Rect(pos, (1, 1)), 'width': 3}
        reds = ((255, 0, 0), (255, 0, 0, 255), surface.map_rgb(red_color), red_color)
        for color in reds:
            surface.fill(surface_color)
            kwargs['color'] = color
            if isinstance(color, int):
                expected_color = surface.unmap_rgb(color)
            else:
                expected_color = red_color
            bounds_rect = self.draw_rect(**kwargs)
            self.assertEqual(surface.get_at(pos), expected_color)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_rect__invalid_color_formats(self):
        """Ensures draw rect handles invalid color formats correctly."""
        pos = (1, 1)
        surface = pygame.Surface((3, 4))
        kwargs = {'surface': surface, 'color': None, 'rect': pygame.Rect(pos, (1, 1)), 'width': 1}
        for expected_color in (2.3, self):
            kwargs['color'] = expected_color
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_rect(**kwargs)

    def test_rect__fill(self):
        self.surf_w, self.surf_h = self.surf_size = (320, 200)
        self.surf = pygame.Surface(self.surf_size, pygame.SRCALPHA)
        self.color = (1, 13, 24, 205)
        rect = pygame.Rect(10, 10, 25, 20)
        drawn = self.draw_rect(self.surf, self.color, rect, 0)
        self.assertEqual(drawn, rect)
        for pt in test_utils.rect_area_pts(rect):
            color_at_pt = self.surf.get_at(pt)
            self.assertEqual(color_at_pt, self.color)
        for pt in test_utils.rect_outer_bounds(rect):
            color_at_pt = self.surf.get_at(pt)
            self.assertNotEqual(color_at_pt, self.color)
        bgcolor = pygame.Color('black')
        self.surf.fill(bgcolor)
        hrect = pygame.Rect(1, 1, self.surf_w - 2, 1)
        vrect = pygame.Rect(1, 3, 1, self.surf_h - 4)
        drawn = self.draw_rect(self.surf, self.color, hrect, 0)
        self.assertEqual(drawn, hrect)
        x, y = hrect.topleft
        w, h = hrect.size
        self.assertEqual(self.surf.get_at((x - 1, y)), bgcolor)
        self.assertEqual(self.surf.get_at((x + w, y)), bgcolor)
        for i in range(x, x + w):
            self.assertEqual(self.surf.get_at((i, y)), self.color)
        drawn = self.draw_rect(self.surf, self.color, vrect, 0)
        self.assertEqual(drawn, vrect)
        x, y = vrect.topleft
        w, h = vrect.size
        self.assertEqual(self.surf.get_at((x, y - 1)), bgcolor)
        self.assertEqual(self.surf.get_at((x, y + h)), bgcolor)
        for i in range(y, y + h):
            self.assertEqual(self.surf.get_at((x, i)), self.color)

    def test_rect__one_pixel_lines(self):
        self.surf = pygame.Surface((320, 200), pygame.SRCALPHA)
        self.color = (1, 13, 24, 205)
        rect = pygame.Rect(10, 10, 56, 20)
        drawn = self.draw_rect(self.surf, self.color, rect, 1)
        self.assertEqual(drawn, rect)
        for pt in test_utils.rect_perimeter_pts(drawn):
            color_at_pt = self.surf.get_at(pt)
            self.assertEqual(color_at_pt, self.color)
        for pt in test_utils.rect_outer_bounds(drawn):
            color_at_pt = self.surf.get_at(pt)
            self.assertNotEqual(color_at_pt, self.color)

    def test_rect__draw_line_width(self):
        surface = pygame.Surface((100, 100))
        surface.fill('black')
        color = pygame.Color(255, 255, 255)
        rect_width = 80
        rect_height = 50
        line_width = 10
        pygame.draw.rect(surface, color, pygame.Rect(0, 0, rect_width, rect_height), line_width)
        for i in range(line_width):
            self.assertEqual(surface.get_at((i, i)), color)
            self.assertEqual(surface.get_at((rect_width - i - 1, i)), color)
            self.assertEqual(surface.get_at((i, rect_height - i - 1)), color)
            self.assertEqual(surface.get_at((rect_width - i - 1, rect_height - i - 1)), color)
        self.assertEqual(surface.get_at((line_width, line_width)), (0, 0, 0))
        self.assertEqual(surface.get_at((rect_width - line_width - 1, line_width)), (0, 0, 0))
        self.assertEqual(surface.get_at((line_width, rect_height - line_width - 1)), (0, 0, 0))
        self.assertEqual(surface.get_at((rect_width - line_width - 1, rect_height - line_width - 1)), (0, 0, 0))

    def test_rect__bounding_rect(self):
        """Ensures draw rect returns the correct bounding rect.

        Tests rects on and off the surface and a range of width/thickness
        values.
        """
        rect_color = pygame.Color('red')
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
                    rect = pygame.Rect((0, 0), (width, height))
                    setattr(rect, attr, pos)
                    for thickness in range(4):
                        surface.fill(surf_color)
                        bounding_rect = self.draw_rect(surface, rect_color, rect, thickness)
                        expected_rect = create_bounding_rect(surface, surf_color, rect.topleft)
                        self.assertEqual(bounding_rect, expected_rect, f'thickness={thickness}')

    def test_rect__surface_clip(self):
        """Ensures draw rect respects a surface's clip area.

        Tests drawing the rect filled and unfilled.
        """
        surfw = surfh = 30
        rect_color = pygame.Color('red')
        surface_color = pygame.Color('green')
        surface = pygame.Surface((surfw, surfh))
        surface.fill(surface_color)
        clip_rect = pygame.Rect((0, 0), (8, 10))
        clip_rect.center = surface.get_rect().center
        test_rect = clip_rect.copy()
        for width in (0, 1):
            for center in rect_corners_mids_and_center(clip_rect):
                test_rect.center = center
                surface.set_clip(None)
                surface.fill(surface_color)
                self.draw_rect(surface, rect_color, test_rect, width)
                expected_pts = get_color_points(surface, rect_color, clip_rect)
                surface.fill(surface_color)
                surface.set_clip(clip_rect)
                self.draw_rect(surface, rect_color, test_rect, width)
                surface.lock()
                for pt in ((x, y) for x in range(surfw) for y in range(surfh)):
                    if pt in expected_pts:
                        expected_color = rect_color
                    else:
                        expected_color = surface_color
                    self.assertEqual(surface.get_at(pt), expected_color, pt)
                surface.unlock()