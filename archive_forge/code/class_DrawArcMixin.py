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
class DrawArcMixin:
    """Mixin tests for drawing arcs.

    This class contains all the general arc drawing tests.
    """

    def test_arc__args(self):
        """Ensures draw arc accepts the correct args."""
        bounds_rect = self.draw_arc(pygame.Surface((3, 3)), (0, 10, 0, 50), (1, 1, 2, 2), 0, 1, 1)
        self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_arc__args_without_width(self):
        """Ensures draw arc accepts the args without a width."""
        bounds_rect = self.draw_arc(pygame.Surface((2, 2)), (1, 1, 1, 99), pygame.Rect((0, 0), (2, 2)), 1.1, 2.1)
        self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_arc__args_with_negative_width(self):
        """Ensures draw arc accepts the args with negative width."""
        bounds_rect = self.draw_arc(pygame.Surface((3, 3)), (10, 10, 50, 50), (1, 1, 2, 2), 0, 1, -1)
        self.assertIsInstance(bounds_rect, pygame.Rect)
        self.assertEqual(bounds_rect, pygame.Rect(1, 1, 0, 0))

    def test_arc__args_with_width_gt_radius(self):
        """Ensures draw arc accepts the args with
        width > rect.w // 2 and width > rect.h // 2.
        """
        rect = pygame.Rect((0, 0), (4, 4))
        bounds_rect = self.draw_arc(pygame.Surface((3, 3)), (10, 10, 50, 50), rect, 0, 45, rect.w // 2 + 1)
        self.assertIsInstance(bounds_rect, pygame.Rect)
        bounds_rect = self.draw_arc(pygame.Surface((3, 3)), (10, 10, 50, 50), rect, 0, 45, rect.h // 2 + 1)
        self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_arc__kwargs(self):
        """Ensures draw arc accepts the correct kwargs
        with and without a width arg.
        """
        kwargs_list = [{'surface': pygame.Surface((4, 4)), 'color': pygame.Color('yellow'), 'rect': pygame.Rect((0, 0), (3, 2)), 'start_angle': 0.5, 'stop_angle': 3, 'width': 1}, {'surface': pygame.Surface((2, 1)), 'color': (0, 10, 20), 'rect': (0, 0, 2, 2), 'start_angle': 1, 'stop_angle': 3.1}]
        for kwargs in kwargs_list:
            bounds_rect = self.draw_arc(**kwargs)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_arc__kwargs_order_independent(self):
        """Ensures draw arc's kwargs are not order dependent."""
        bounds_rect = self.draw_arc(stop_angle=1, start_angle=2.2, color=(1, 2, 3), surface=pygame.Surface((3, 2)), width=1, rect=pygame.Rect((1, 0), (2, 3)))
        self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_arc__args_missing(self):
        """Ensures draw arc detects any missing required args."""
        surface = pygame.Surface((1, 1))
        color = pygame.Color('red')
        rect = pygame.Rect((0, 0), (2, 2))
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_arc(surface, color, rect, 0.1)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_arc(surface, color, rect)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_arc(surface, color)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_arc(surface)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_arc()

    def test_arc__kwargs_missing(self):
        """Ensures draw arc detects any missing required kwargs."""
        kwargs = {'surface': pygame.Surface((1, 2)), 'color': pygame.Color('red'), 'rect': pygame.Rect((1, 0), (2, 2)), 'start_angle': 0.1, 'stop_angle': 2, 'width': 1}
        for name in ('stop_angle', 'start_angle', 'rect', 'color', 'surface'):
            invalid_kwargs = dict(kwargs)
            invalid_kwargs.pop(name)
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_arc(**invalid_kwargs)

    def test_arc__arg_invalid_types(self):
        """Ensures draw arc detects invalid arg types."""
        surface = pygame.Surface((2, 2))
        color = pygame.Color('blue')
        rect = pygame.Rect((1, 1), (3, 3))
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_arc(surface, color, rect, 0, 1, '1')
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_arc(surface, color, rect, 0, '1', 1)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_arc(surface, color, rect, '1', 0, 1)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_arc(surface, color, (1, 2, 3, 4, 5), 0, 1, 1)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_arc(surface, 2.3, rect, 0, 1, 1)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_arc(rect, color, rect, 0, 1, 1)

    def test_arc__kwarg_invalid_types(self):
        """Ensures draw arc detects invalid kwarg types."""
        surface = pygame.Surface((3, 3))
        color = pygame.Color('green')
        rect = pygame.Rect((0, 1), (4, 2))
        start = 3
        stop = 4
        kwargs_list = [{'surface': pygame.Surface, 'color': color, 'rect': rect, 'start_angle': start, 'stop_angle': stop, 'width': 1}, {'surface': surface, 'color': 2.3, 'rect': rect, 'start_angle': start, 'stop_angle': stop, 'width': 1}, {'surface': surface, 'color': color, 'rect': (0, 0, 0), 'start_angle': start, 'stop_angle': stop, 'width': 1}, {'surface': surface, 'color': color, 'rect': rect, 'start_angle': '1', 'stop_angle': stop, 'width': 1}, {'surface': surface, 'color': color, 'rect': rect, 'start_angle': start, 'stop_angle': '1', 'width': 1}, {'surface': surface, 'color': color, 'rect': rect, 'start_angle': start, 'stop_angle': stop, 'width': 1.1}]
        for kwargs in kwargs_list:
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_arc(**kwargs)

    def test_arc__kwarg_invalid_name(self):
        """Ensures draw arc detects invalid kwarg names."""
        surface = pygame.Surface((2, 3))
        color = pygame.Color('cyan')
        rect = pygame.Rect((0, 1), (2, 2))
        start = 0.9
        stop = 2.3
        kwargs_list = [{'surface': surface, 'color': color, 'rect': rect, 'start_angle': start, 'stop_angle': stop, 'width': 1, 'invalid': 1}, {'surface': surface, 'color': color, 'rect': rect, 'start_angle': start, 'stop_angle': stop, 'invalid': 1}]
        for kwargs in kwargs_list:
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_arc(**kwargs)

    def test_arc__args_and_kwargs(self):
        """Ensures draw arc accepts a combination of args/kwargs"""
        surface = pygame.Surface((3, 1))
        color = (255, 255, 0, 0)
        rect = pygame.Rect((1, 0), (2, 3))
        start = 0.6
        stop = 2
        width = 1
        kwargs = {'surface': surface, 'color': color, 'rect': rect, 'start_angle': start, 'stop_angle': stop, 'width': width}
        for name in ('surface', 'color', 'rect', 'start_angle', 'stop_angle'):
            kwargs.pop(name)
            if 'surface' == name:
                bounds_rect = self.draw_arc(surface, **kwargs)
            elif 'color' == name:
                bounds_rect = self.draw_arc(surface, color, **kwargs)
            elif 'rect' == name:
                bounds_rect = self.draw_arc(surface, color, rect, **kwargs)
            elif 'start_angle' == name:
                bounds_rect = self.draw_arc(surface, color, rect, start, **kwargs)
            elif 'stop_angle' == name:
                bounds_rect = self.draw_arc(surface, color, rect, start, stop, **kwargs)
            else:
                bounds_rect = self.draw_arc(surface, color, rect, start, stop, width, **kwargs)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_arc__valid_width_values(self):
        """Ensures draw arc accepts different width values."""
        arc_color = pygame.Color('yellow')
        surface_color = pygame.Color('white')
        surface = pygame.Surface((6, 6))
        rect = pygame.Rect((0, 0), (4, 4))
        rect.center = surface.get_rect().center
        pos = (rect.centerx + 1, rect.centery + 1)
        kwargs = {'surface': surface, 'color': arc_color, 'rect': rect, 'start_angle': 0, 'stop_angle': 7, 'width': None}
        for width in (-50, -10, -3, -2, -1, 0, 1, 2, 3, 10, 50):
            msg = f'width={width}'
            surface.fill(surface_color)
            kwargs['width'] = width
            expected_color = arc_color if width > 0 else surface_color
            bounds_rect = self.draw_arc(**kwargs)
            self.assertEqual(surface.get_at(pos), expected_color, msg)
            self.assertIsInstance(bounds_rect, pygame.Rect, msg)

    def test_arc__valid_stop_angle_values(self):
        """Ensures draw arc accepts different stop_angle values."""
        expected_color = pygame.Color('blue')
        surface_color = pygame.Color('white')
        surface = pygame.Surface((6, 6))
        rect = pygame.Rect((0, 0), (4, 4))
        rect.center = surface.get_rect().center
        pos = (rect.centerx, rect.centery + 1)
        kwargs = {'surface': surface, 'color': expected_color, 'rect': rect, 'start_angle': -17, 'stop_angle': None, 'width': 1}
        for stop_angle in (-10, -5.5, -1, 0, 1, 5.5, 10):
            msg = f'stop_angle={stop_angle}'
            surface.fill(surface_color)
            kwargs['stop_angle'] = stop_angle
            bounds_rect = self.draw_arc(**kwargs)
            self.assertEqual(surface.get_at(pos), expected_color, msg)
            self.assertIsInstance(bounds_rect, pygame.Rect, msg)

    def test_arc__valid_start_angle_values(self):
        """Ensures draw arc accepts different start_angle values."""
        expected_color = pygame.Color('blue')
        surface_color = pygame.Color('white')
        surface = pygame.Surface((6, 6))
        rect = pygame.Rect((0, 0), (4, 4))
        rect.center = surface.get_rect().center
        pos = (rect.centerx + 1, rect.centery + 1)
        kwargs = {'surface': surface, 'color': expected_color, 'rect': rect, 'start_angle': None, 'stop_angle': 17, 'width': 1}
        for start_angle in (-10.0, -5.5, -1, 0, 1, 5.5, 10.0):
            msg = f'start_angle={start_angle}'
            surface.fill(surface_color)
            kwargs['start_angle'] = start_angle
            bounds_rect = self.draw_arc(**kwargs)
            self.assertEqual(surface.get_at(pos), expected_color, msg)
            self.assertIsInstance(bounds_rect, pygame.Rect, msg)

    def test_arc__valid_rect_formats(self):
        """Ensures draw arc accepts different rect formats."""
        expected_color = pygame.Color('red')
        surface_color = pygame.Color('black')
        surface = pygame.Surface((6, 6))
        rect = pygame.Rect((0, 0), (4, 4))
        rect.center = surface.get_rect().center
        pos = (rect.centerx + 1, rect.centery + 1)
        kwargs = {'surface': surface, 'color': expected_color, 'rect': None, 'start_angle': 0, 'stop_angle': 7, 'width': 1}
        rects = (rect, (rect.topleft, rect.size), (rect.x, rect.y, rect.w, rect.h))
        for rect in rects:
            surface.fill(surface_color)
            kwargs['rect'] = rect
            bounds_rect = self.draw_arc(**kwargs)
            self.assertEqual(surface.get_at(pos), expected_color)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_arc__valid_color_formats(self):
        """Ensures draw arc accepts different color formats."""
        green_color = pygame.Color('green')
        surface_color = pygame.Color('black')
        surface = pygame.Surface((6, 6))
        rect = pygame.Rect((0, 0), (4, 4))
        rect.center = surface.get_rect().center
        pos = (rect.centerx + 1, rect.centery + 1)
        kwargs = {'surface': surface, 'color': None, 'rect': rect, 'start_angle': 0, 'stop_angle': 7, 'width': 1}
        greens = ((0, 255, 0), (0, 255, 0, 255), surface.map_rgb(green_color), green_color)
        for color in greens:
            surface.fill(surface_color)
            kwargs['color'] = color
            if isinstance(color, int):
                expected_color = surface.unmap_rgb(color)
            else:
                expected_color = green_color
            bounds_rect = self.draw_arc(**kwargs)
            self.assertEqual(surface.get_at(pos), expected_color)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_arc__invalid_color_formats(self):
        """Ensures draw arc handles invalid color formats correctly."""
        pos = (1, 1)
        surface = pygame.Surface((4, 3))
        kwargs = {'surface': surface, 'color': None, 'rect': pygame.Rect(pos, (2, 2)), 'start_angle': 5, 'stop_angle': 6.1, 'width': 1}
        for expected_color in (2.3, self):
            kwargs['color'] = expected_color
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_arc(**kwargs)

    def test_arc(self):
        """Ensure draw arc works correctly."""
        black = pygame.Color('black')
        red = pygame.Color('red')
        surface = pygame.Surface((100, 150))
        surface.fill(black)
        rect = (0, 0, 80, 40)
        start_angle = 0.0
        stop_angle = 3.14
        width = 3
        pygame.draw.arc(surface, red, rect, start_angle, stop_angle, width)
        pygame.image.save(surface, 'arc.png')
        x = 20
        for y in range(2, 5):
            self.assertEqual(surface.get_at((x, y)), red)
        self.assertEqual(surface.get_at((0, 0)), black)

    def test_arc__bounding_rect(self):
        """Ensures draw arc returns the correct bounding rect.

        Tests arcs on and off the surface and a range of width/thickness
        values.
        """
        arc_color = pygame.Color('red')
        surf_color = pygame.Color('black')
        min_width = min_height = 5
        max_width = max_height = 7
        sizes = ((min_width, min_height), (max_width, max_height))
        surface = pygame.Surface((20, 20), 0, 32)
        surf_rect = surface.get_rect()
        big_rect = surf_rect.inflate(min_width * 2 + 1, min_height * 2 + 1)
        start_angle = 0
        stop_angles = (0, 2, 3, 5, math.ceil(2 * math.pi))
        for pos in rect_corners_mids_and_center(surf_rect) + rect_corners_mids_and_center(big_rect):
            for attr in RECT_POSITION_ATTRIBUTES:
                for width, height in sizes:
                    arc_rect = pygame.Rect((0, 0), (width, height))
                    setattr(arc_rect, attr, pos)
                    for thickness in (0, 1, 2, 3, min(width, height)):
                        for stop_angle in stop_angles:
                            surface.fill(surf_color)
                            bounding_rect = self.draw_arc(surface, arc_color, arc_rect, start_angle, stop_angle, thickness)
                            expected_rect = create_bounding_rect(surface, surf_color, arc_rect.topleft)
                            self.assertEqual(bounding_rect, expected_rect, f'thickness={thickness}')

    def test_arc__surface_clip(self):
        """Ensures draw arc respects a surface's clip area."""
        surfw = surfh = 30
        start = 0.1
        end = 0
        arc_color = pygame.Color('red')
        surface_color = pygame.Color('green')
        surface = pygame.Surface((surfw, surfh))
        surface.fill(surface_color)
        clip_rect = pygame.Rect((0, 0), (11, 11))
        clip_rect.center = surface.get_rect().center
        pos_rect = clip_rect.copy()
        for thickness in (1, 3):
            for center in rect_corners_mids_and_center(clip_rect):
                pos_rect.center = center
                surface.set_clip(None)
                surface.fill(surface_color)
                self.draw_arc(surface, arc_color, pos_rect, start, end, thickness)
                expected_pts = get_color_points(surface, arc_color, clip_rect)
                surface.fill(surface_color)
                surface.set_clip(clip_rect)
                self.draw_arc(surface, arc_color, pos_rect, start, end, thickness)
                surface.lock()
                for pt in ((x, y) for x in range(surfw) for y in range(surfh)):
                    if pt in expected_pts:
                        expected_color = arc_color
                    else:
                        expected_color = surface_color
                    self.assertEqual(surface.get_at(pt), expected_color, pt)
                surface.unlock()