from the Sprite and AbstractGroup classes below, it is strongly recommended
from weakref import WeakSet
from warnings import warn
import pygame
from pygame.rect import Rect
from pygame.time import get_ticks
from pygame.mask import from_surface
def _set_sprite(self, sprite):
    self.add_internal(sprite)
    sprite.add_internal(self)
    return sprite