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
class PythonDrawTestCase(unittest.TestCase):
    """Base class to test draw_py module functions."""
    draw_polygon = staticmethod(draw_py.draw_polygon)
    draw_line = staticmethod(draw_py.draw_line)
    draw_lines = staticmethod(draw_py.draw_lines)
    draw_aaline = staticmethod(draw_py.draw_aaline)
    draw_aalines = staticmethod(draw_py.draw_aalines)