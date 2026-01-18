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
def _check_1_pixel_sized_ellipse(self, surface, collide_rect, surface_color, ellipse_color):
    surf_w, surf_h = surface.get_size()
    surface.lock()
    for pos in ((x, y) for y in range(surf_h) for x in range(surf_w)):
        if collide_rect.collidepoint(pos):
            expected_color = ellipse_color
        else:
            expected_color = surface_color
        self.assertEqual(surface.get_at(pos), expected_color, f'collide_rect={collide_rect}, pos={pos}')
    surface.unlock()