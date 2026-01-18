import platform
import unittest
import pygame
from pygame.locals import *
from pygame.pixelcopy import surface_to_array, map_array, array_to_surface, make_surface
def _fill_surface(self, surf, palette=None):
    if palette is None:
        palette = self.test_palette
    surf.fill(palette[1], (0, 0, 5, 6))
    surf.fill(palette[2], (5, 0, 5, 6))
    surf.fill(palette[3], (0, 6, 5, 6))
    surf.fill(palette[4], (5, 6, 5, 6))