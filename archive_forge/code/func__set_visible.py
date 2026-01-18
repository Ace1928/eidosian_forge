from the Sprite and AbstractGroup classes below, it is strongly recommended
from weakref import WeakSet
from warnings import warn
import pygame
from pygame.rect import Rect
from pygame.time import get_ticks
from pygame.mask import from_surface
def _set_visible(self, val):
    """set the visible value (0 or 1) and makes the sprite dirty"""
    self._visible = val
    if self.dirty < 2:
        self.dirty = 1