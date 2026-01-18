import platform
import unittest
import pygame
from pygame.locals import *
from pygame.pixelcopy import surface_to_array, map_array, array_to_surface, make_surface
def _make_src_surface(self, bitsize, srcalpha=False, palette=None):
    surf = self._make_surface(bitsize, srcalpha, palette)
    self._fill_surface(surf, palette)
    return surf