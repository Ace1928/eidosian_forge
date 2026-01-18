from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def _draw_component_pattern_plus(self, mask, size, pos, inverse=False):
    pattern = pygame.mask.Mask((size, size))
    xmid = ymid = size // 2
    for y in range(size):
        for x in range(size):
            if x == xmid or y == ymid:
                pattern.set_at((x, y))
    if inverse:
        mask.erase(pattern, pos)
        pattern.invert()
    else:
        mask.draw(pattern, pos)
    return pattern