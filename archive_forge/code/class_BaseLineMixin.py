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
class BaseLineMixin:
    """Mixin base for drawing various lines.

    This class contains general helper methods and setup for testing the
    different types of lines.
    """
    COLORS = ((0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 255))

    @staticmethod
    def _create_surfaces():
        surfaces = []
        for size in ((49, 49), (50, 50)):
            for depth in (8, 16, 24, 32):
                for flags in (0, SRCALPHA):
                    surface = pygame.display.set_mode(size, flags, depth)
                    surfaces.append(surface)
                    surfaces.append(surface.convert_alpha())
        return surfaces

    @staticmethod
    def _rect_lines(rect):
        for pt in rect_corners_mids_and_center(rect):
            if pt in [rect.midleft, rect.center]:
                continue
            yield (rect.midleft, pt)
            yield (pt, rect.midleft)