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
@staticmethod
def _rect_lines(rect):
    for pt in rect_corners_mids_and_center(rect):
        if pt in [rect.midleft, rect.center]:
            continue
        yield (rect.midleft, pt)
        yield (pt, rect.midleft)