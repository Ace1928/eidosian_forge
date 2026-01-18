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
class DrawCircleMixin:
    """Mixin tests for drawing circles.

    This class contains all the general circle drawing tests.
    """

    def test_circle__args(self):
        """Ensures draw circle accepts the correct args."""
        bounds_rect = self.draw_circle(pygame.Surface((3, 3)), (0, 10, 0, 50), (0, 0), 3, 1, 1, 0, 1, 1)
        self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_circle__args_without_width(self):
        """Ensures draw circle accepts the args without a width and
        quadrants."""
        bounds_rect = self.draw_circle(pygame.Surface((2, 2)), (0, 0, 0, 50), (1, 1), 1)
        self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_circle__args_with_negative_width(self):
        """Ensures draw circle accepts the args with negative width."""
        bounds_rect = self.draw_circle(pygame.Surface((2, 2)), (0, 0, 0, 50), (1, 1), 1, -1)
        self.assertIsInstance(bounds_rect, pygame.Rect)
        self.assertEqual(bounds_rect, pygame.Rect(1, 1, 0, 0))

    def test_circle__args_with_width_gt_radius(self):
        """Ensures draw circle accepts the args with width > radius."""
        bounds_rect = self.draw_circle(pygame.Surface((2, 2)), (0, 0, 0, 50), (1, 1), 2, 3, 0, 0, 0, 0)
        self.assertIsInstance(bounds_rect, pygame.Rect)
        self.assertEqual(bounds_rect, pygame.Rect(0, 0, 2, 2))

    def test_circle__kwargs(self):
        """Ensures draw circle accepts the correct kwargs
        with and without a width and quadrant arguments.
        """
        kwargs_list = [{'surface': pygame.Surface((4, 4)), 'color': pygame.Color('yellow'), 'center': (2, 2), 'radius': 2, 'width': 1, 'draw_top_right': True, 'draw_top_left': True, 'draw_bottom_left': False, 'draw_bottom_right': True}, {'surface': pygame.Surface((2, 1)), 'color': (0, 10, 20), 'center': (1, 1), 'radius': 1}]
        for kwargs in kwargs_list:
            bounds_rect = self.draw_circle(**kwargs)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_circle__kwargs_order_independent(self):
        """Ensures draw circle's kwargs are not order dependent."""
        bounds_rect = self.draw_circle(draw_top_right=False, color=(10, 20, 30), surface=pygame.Surface((3, 2)), width=0, draw_bottom_left=False, center=(1, 0), draw_bottom_right=False, radius=2, draw_top_left=True)
        self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_circle__args_missing(self):
        """Ensures draw circle detects any missing required args."""
        surface = pygame.Surface((1, 1))
        color = pygame.Color('blue')
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_circle(surface, color, (0, 0))
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_circle(surface, color)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_circle(surface)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_circle()

    def test_circle__kwargs_missing(self):
        """Ensures draw circle detects any missing required kwargs."""
        kwargs = {'surface': pygame.Surface((1, 2)), 'color': pygame.Color('red'), 'center': (1, 0), 'radius': 2, 'width': 1, 'draw_top_right': False, 'draw_top_left': False, 'draw_bottom_left': False, 'draw_bottom_right': True}
        for name in ('radius', 'center', 'color', 'surface'):
            invalid_kwargs = dict(kwargs)
            invalid_kwargs.pop(name)
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_circle(**invalid_kwargs)

    def test_circle__arg_invalid_types(self):
        """Ensures draw circle detects invalid arg types."""
        surface = pygame.Surface((2, 2))
        color = pygame.Color('blue')
        center = (1, 1)
        radius = 1
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_circle(surface, color, center, radius, 1, 'a', 1, 1, 1)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_circle(surface, color, center, radius, 1, 1, 'b', 1, 1)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_circle(surface, color, center, radius, 1, 1, 1, 'c', 1)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_circle(surface, color, center, radius, 1, 1, 1, 1, 'd')
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_circle(surface, color, center, radius, '1')
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_circle(surface, color, center, '2')
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_circle(surface, color, (1, 2, 3), radius)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_circle(surface, 2.3, center, radius)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_circle((1, 2, 3, 4), color, center, radius)

    def test_circle__kwarg_invalid_types(self):
        """Ensures draw circle detects invalid kwarg types."""
        surface = pygame.Surface((3, 3))
        color = pygame.Color('green')
        center = (0, 1)
        radius = 1
        width = 1
        quadrant = 1
        kwargs_list = [{'surface': pygame.Surface, 'color': color, 'center': center, 'radius': radius, 'width': width, 'draw_top_right': True, 'draw_top_left': True, 'draw_bottom_left': True, 'draw_bottom_right': True}, {'surface': surface, 'color': 2.3, 'center': center, 'radius': radius, 'width': width, 'draw_top_right': True, 'draw_top_left': True, 'draw_bottom_left': True, 'draw_bottom_right': True}, {'surface': surface, 'color': color, 'center': (1, 1, 1), 'radius': radius, 'width': width, 'draw_top_right': True, 'draw_top_left': True, 'draw_bottom_left': True, 'draw_bottom_right': True}, {'surface': surface, 'color': color, 'center': center, 'radius': '1', 'width': width, 'draw_top_right': True, 'draw_top_left': True, 'draw_bottom_left': True, 'draw_bottom_right': True}, {'surface': surface, 'color': color, 'center': center, 'radius': radius, 'width': 1.2, 'draw_top_right': True, 'draw_top_left': True, 'draw_bottom_left': True, 'draw_bottom_right': True}, {'surface': surface, 'color': color, 'center': center, 'radius': radius, 'width': width, 'draw_top_right': 'True', 'draw_top_left': True, 'draw_bottom_left': True, 'draw_bottom_right': True}, {'surface': surface, 'color': color, 'center': center, 'radius': radius, 'width': width, 'draw_top_right': True, 'draw_top_left': 'True', 'draw_bottom_left': True, 'draw_bottom_right': True}, {'surface': surface, 'color': color, 'center': center, 'radius': radius, 'width': width, 'draw_top_right': True, 'draw_top_left': True, 'draw_bottom_left': 3.14, 'draw_bottom_right': True}, {'surface': surface, 'color': color, 'center': center, 'radius': radius, 'width': width, 'draw_top_right': True, 'draw_top_left': True, 'draw_bottom_left': True, 'draw_bottom_right': 'quadrant'}]
        for kwargs in kwargs_list:
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_circle(**kwargs)

    def test_circle__kwarg_invalid_name(self):
        """Ensures draw circle detects invalid kwarg names."""
        surface = pygame.Surface((2, 3))
        color = pygame.Color('cyan')
        center = (0, 0)
        radius = 2
        kwargs_list = [{'surface': surface, 'color': color, 'center': center, 'radius': radius, 'width': 1, 'quadrant': 1, 'draw_top_right': True, 'draw_top_left': True, 'draw_bottom_left': True, 'draw_bottom_right': True}, {'surface': surface, 'color': color, 'center': center, 'radius': radius, 'invalid': 1}]
        for kwargs in kwargs_list:
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_circle(**kwargs)

    def test_circle__args_and_kwargs(self):
        """Ensures draw circle accepts a combination of args/kwargs"""
        surface = pygame.Surface((3, 1))
        color = (255, 255, 0, 0)
        center = (1, 0)
        radius = 2
        width = 0
        draw_top_right = True
        draw_top_left = False
        draw_bottom_left = False
        draw_bottom_right = True
        kwargs = {'surface': surface, 'color': color, 'center': center, 'radius': radius, 'width': width, 'draw_top_right': True, 'draw_top_left': True, 'draw_bottom_left': True, 'draw_bottom_right': True}
        for name in ('surface', 'color', 'center', 'radius', 'width', 'draw_top_right', 'draw_top_left', 'draw_bottom_left', 'draw_bottom_right'):
            kwargs.pop(name)
            if 'surface' == name:
                bounds_rect = self.draw_circle(surface, **kwargs)
            elif 'color' == name:
                bounds_rect = self.draw_circle(surface, color, **kwargs)
            elif 'center' == name:
                bounds_rect = self.draw_circle(surface, color, center, **kwargs)
            elif 'radius' == name:
                bounds_rect = self.draw_circle(surface, color, center, radius, **kwargs)
            elif 'width' == name:
                bounds_rect = self.draw_circle(surface, color, center, radius, width, **kwargs)
            elif 'draw_top_right' == name:
                bounds_rect = self.draw_circle(surface, color, center, radius, width, draw_top_right, **kwargs)
            elif 'draw_top_left' == name:
                bounds_rect = self.draw_circle(surface, color, center, radius, width, draw_top_right, draw_top_left, **kwargs)
            elif 'draw_bottom_left' == name:
                bounds_rect = self.draw_circle(surface, color, center, radius, width, draw_top_right, draw_top_left, draw_bottom_left, **kwargs)
            else:
                bounds_rect = self.draw_circle(surface, color, center, radius, width, draw_top_right, draw_top_left, draw_bottom_left, draw_bottom_right, **kwargs)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_circle__valid_width_values(self):
        """Ensures draw circle accepts different width values."""
        center = (2, 2)
        radius = 1
        pos = (center[0] - radius, center[1])
        surface_color = pygame.Color('white')
        surface = pygame.Surface((3, 4))
        color = (10, 20, 30, 255)
        kwargs = {'surface': surface, 'color': color, 'center': center, 'radius': radius, 'width': None, 'draw_top_right': True, 'draw_top_left': True, 'draw_bottom_left': True, 'draw_bottom_right': True}
        for width in (-100, -10, -1, 0, 1, 10, 100):
            surface.fill(surface_color)
            kwargs['width'] = width
            expected_color = color if width >= 0 else surface_color
            bounds_rect = self.draw_circle(**kwargs)
            self.assertEqual(surface.get_at(pos), expected_color)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_circle__valid_radius_values(self):
        """Ensures draw circle accepts different radius values."""
        pos = center = (2, 2)
        surface_color = pygame.Color('white')
        surface = pygame.Surface((3, 4))
        color = (10, 20, 30, 255)
        kwargs = {'surface': surface, 'color': color, 'center': center, 'radius': None, 'width': 0, 'draw_top_right': True, 'draw_top_left': True, 'draw_bottom_left': True, 'draw_bottom_right': True}
        for radius in (-10, -1, 0, 1, 10):
            surface.fill(surface_color)
            kwargs['radius'] = radius
            expected_color = color if radius > 0 else surface_color
            bounds_rect = self.draw_circle(**kwargs)
            self.assertEqual(surface.get_at(pos), expected_color)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_circle__valid_center_formats(self):
        """Ensures draw circle accepts different center formats."""
        expected_color = pygame.Color('red')
        surface_color = pygame.Color('black')
        surface = pygame.Surface((4, 4))
        kwargs = {'surface': surface, 'color': expected_color, 'center': None, 'radius': 1, 'width': 0, 'draw_top_right': True, 'draw_top_left': True, 'draw_bottom_left': True, 'draw_bottom_right': True}
        x, y = (2, 2)
        for center in ((x, y), (x + 0.1, y), (x, y + 0.1), (x + 0.1, y + 0.1)):
            for seq_type in (tuple, list, Vector2):
                surface.fill(surface_color)
                kwargs['center'] = seq_type(center)
                bounds_rect = self.draw_circle(**kwargs)
                self.assertEqual(surface.get_at((x, y)), expected_color)
                self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_circle__valid_color_formats(self):
        """Ensures draw circle accepts different color formats."""
        center = (2, 2)
        radius = 1
        pos = (center[0] - radius, center[1])
        green_color = pygame.Color('green')
        surface_color = pygame.Color('black')
        surface = pygame.Surface((3, 4))
        kwargs = {'surface': surface, 'color': None, 'center': center, 'radius': radius, 'width': 0, 'draw_top_right': True, 'draw_top_left': True, 'draw_bottom_left': True, 'draw_bottom_right': True}
        greens = ((0, 255, 0), (0, 255, 0, 255), surface.map_rgb(green_color), green_color)
        for color in greens:
            surface.fill(surface_color)
            kwargs['color'] = color
            if isinstance(color, int):
                expected_color = surface.unmap_rgb(color)
            else:
                expected_color = green_color
            bounds_rect = self.draw_circle(**kwargs)
            self.assertEqual(surface.get_at(pos), expected_color)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_circle__invalid_color_formats(self):
        """Ensures draw circle handles invalid color formats correctly."""
        kwargs = {'surface': pygame.Surface((4, 3)), 'color': None, 'center': (1, 2), 'radius': 1, 'width': 0, 'draw_top_right': True, 'draw_top_left': True, 'draw_bottom_left': True, 'draw_bottom_right': True}
        for expected_color in (2.3, self):
            kwargs['color'] = expected_color
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_circle(**kwargs)

    def test_circle__floats(self):
        """Ensure that floats are accepted."""
        draw.circle(surface=pygame.Surface((4, 4)), color=(255, 255, 127), center=(1.5, 1.5), radius=1.3, width=0, draw_top_right=True, draw_top_left=True, draw_bottom_left=True, draw_bottom_right=True)
        draw.circle(surface=pygame.Surface((4, 4)), color=(255, 255, 127), center=Vector2(1.5, 1.5), radius=1.3, width=0, draw_top_right=True, draw_top_left=True, draw_bottom_left=True, draw_bottom_right=True)
        draw.circle(pygame.Surface((2, 2)), (0, 0, 0, 50), (1.3, 1.3), 1.2)

    def test_circle__bounding_rect(self):
        """Ensures draw circle returns the correct bounding rect.

        Tests circles on and off the surface and a range of width/thickness
        values.
        """
        circle_color = pygame.Color('red')
        surf_color = pygame.Color('black')
        max_radius = 3
        surface = pygame.Surface((30, 30), 0, 32)
        surf_rect = surface.get_rect()
        big_rect = surf_rect.inflate(max_radius * 2 - 1, max_radius * 2 - 1)
        for pos in rect_corners_mids_and_center(surf_rect) + rect_corners_mids_and_center(big_rect):
            for radius in range(max_radius + 1):
                for thickness in range(radius + 1):
                    surface.fill(surf_color)
                    bounding_rect = self.draw_circle(surface, circle_color, pos, radius, thickness)
                    expected_rect = create_bounding_rect(surface, surf_color, pos)
                    with self.subTest(surface=surface, circle_color=circle_color, pos=pos, radius=radius, thickness=thickness):
                        self.assertEqual(bounding_rect, expected_rect)

    def test_circle_negative_radius(self):
        """Ensures negative radius circles return zero sized bounding rect."""
        surf = pygame.Surface((200, 200))
        color = (0, 0, 0, 50)
        center = (surf.get_height() // 2, surf.get_height() // 2)
        bounding_rect = self.draw_circle(surf, color, center, radius=-1, width=1)
        self.assertEqual(bounding_rect.size, (0, 0))

    def test_circle_zero_radius(self):
        """Ensures zero radius circles does not draw a center pixel.

        NOTE: This is backwards incompatible behaviour with 1.9.x.
        """
        surf = pygame.Surface((200, 200))
        circle_color = pygame.Color('red')
        surf_color = pygame.Color('black')
        center = (100, 100)
        radius = 0
        width = 1
        bounding_rect = self.draw_circle(surf, circle_color, center, radius, width)
        expected_rect = create_bounding_rect(surf, surf_color, center)
        self.assertEqual(bounding_rect, expected_rect)
        self.assertEqual(bounding_rect, pygame.Rect(100, 100, 0, 0))

    def test_circle__surface_clip(self):
        """Ensures draw circle respects a surface's clip area.

        Tests drawing the circle filled and unfilled.
        """
        surfw = surfh = 25
        circle_color = pygame.Color('red')
        surface_color = pygame.Color('green')
        surface = pygame.Surface((surfw, surfh))
        surface.fill(surface_color)
        clip_rect = pygame.Rect((0, 0), (10, 10))
        clip_rect.center = surface.get_rect().center
        radius = clip_rect.w // 2 + 1
        for width in (0, 1):
            for center in rect_corners_mids_and_center(clip_rect):
                surface.set_clip(None)
                surface.fill(surface_color)
                self.draw_circle(surface, circle_color, center, radius, width)
                expected_pts = get_color_points(surface, circle_color, clip_rect)
                surface.fill(surface_color)
                surface.set_clip(clip_rect)
                self.draw_circle(surface, circle_color, center, radius, width)
                surface.lock()
                for pt in ((x, y) for x in range(surfw) for y in range(surfh)):
                    if pt in expected_pts:
                        expected_color = circle_color
                    else:
                        expected_color = surface_color
                    self.assertEqual(surface.get_at(pt), expected_color, pt)
                surface.unlock()

    def test_circle_shape(self):
        """Ensures there are no holes in the circle, and no overdrawing.

        Tests drawing a thick circle.
        Measures the distance of the drawn pixels from the circle center.
        """
        surfw = surfh = 100
        circle_color = pygame.Color('red')
        surface_color = pygame.Color('green')
        surface = pygame.Surface((surfw, surfh))
        surface.fill(surface_color)
        cx, cy = center = (50, 50)
        radius = 45
        width = 25
        dest_rect = self.draw_circle(surface, circle_color, center, radius, width)
        for pt in test_utils.rect_area_pts(dest_rect):
            x, y = pt
            sqr_distance = (x - cx) ** 2 + (y - cy) ** 2
            if (radius - width + 1) ** 2 < sqr_distance < (radius - 1) ** 2:
                self.assertEqual(surface.get_at(pt), circle_color)
            if sqr_distance < (radius - width - 1) ** 2 or sqr_distance > (radius + 1) ** 2:
                self.assertEqual(surface.get_at(pt), surface_color)

    def test_circle__diameter(self):
        """Ensures draw circle is twice size of radius high and wide."""
        surf = pygame.Surface((200, 200))
        color = (0, 0, 0, 50)
        center = (surf.get_height() // 2, surf.get_height() // 2)
        width = 1
        radius = 6
        for radius in range(1, 65):
            bounding_rect = self.draw_circle(surf, color, center, radius, width)
            self.assertEqual(bounding_rect.width, radius * 2)
            self.assertEqual(bounding_rect.height, radius * 2)

    def test_x_bounds(self):
        """ensures a circle is drawn properly when there is a negative x, or a big x."""
        surf = pygame.Surface((200, 200))
        bgcolor = (0, 0, 0, 255)
        surf.fill(bgcolor)
        color = (255, 0, 0, 255)
        width = 1
        radius = 10
        where = (0, 30)
        bounding_rect1 = self.draw_circle(surf, color, where, radius=radius)
        self.assertEqual(bounding_rect1, pygame.Rect(0, where[1] - radius, where[0] + radius, radius * 2))
        self.assertEqual(surf.get_at((where[0] if where[0] > 0 else 0, where[1])), color)
        self.assertEqual(surf.get_at((where[0] + radius + 1, where[1])), bgcolor)
        self.assertEqual(surf.get_at((where[0] + radius - 1, where[1])), color)
        surf.fill(bgcolor)
        where = (-1e+30, 80)
        bounding_rect1 = self.draw_circle(surf, color, where, radius=radius)
        self.assertEqual(bounding_rect1, pygame.Rect(where[0], where[1], 0, 0))
        self.assertEqual(surf.get_at((0 + radius, where[1])), bgcolor)
        surf.fill(bgcolor)
        where = (surf.get_width() + radius * 2, 80)
        bounding_rect1 = self.draw_circle(surf, color, where, radius=radius)
        self.assertEqual(bounding_rect1, pygame.Rect(where[0], where[1], 0, 0))
        self.assertEqual(surf.get_at((0, where[1])), bgcolor)
        self.assertEqual(surf.get_at((0 + radius // 2, where[1])), bgcolor)
        self.assertEqual(surf.get_at((surf.get_width() - 1, where[1])), bgcolor)
        self.assertEqual(surf.get_at((surf.get_width() - radius, where[1])), bgcolor)
        surf.fill(bgcolor)
        where = (-1, 80)
        bounding_rect1 = self.draw_circle(surf, color, where, radius=radius)
        self.assertEqual(bounding_rect1, pygame.Rect(0, where[1] - radius, where[0] + radius, radius * 2))
        self.assertEqual(surf.get_at((where[0] if where[0] > 0 else 0, where[1])), color)
        self.assertEqual(surf.get_at((where[0] + radius, where[1])), bgcolor)
        self.assertEqual(surf.get_at((where[0] + radius - 1, where[1])), color)