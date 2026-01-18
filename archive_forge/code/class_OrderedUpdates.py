from the Sprite and AbstractGroup classes below, it is strongly recommended
from weakref import WeakSet
from warnings import warn
import pygame
from pygame.rect import Rect
from pygame.time import get_ticks
from pygame.mask import from_surface
class OrderedUpdates(RenderUpdates):
    """RenderUpdates class that draws Sprites in order of addition

    pygame.sprite.OrderedUpdates(*sprites): return OrderedUpdates

    This class derives from pygame.sprite.RenderUpdates().  It maintains
    the order in which the Sprites were added to the Group for rendering.
    This makes adding and removing Sprites from the Group a little
    slower than regular Groups.

    """

    def __init__(self, *sprites):
        self._spritelist = []
        RenderUpdates.__init__(self, *sprites)

    def sprites(self):
        return self._spritelist.copy()

    def add_internal(self, sprite, layer=None):
        RenderUpdates.add_internal(self, sprite)
        self._spritelist.append(sprite)

    def remove_internal(self, sprite):
        RenderUpdates.remove_internal(self, sprite)
        self._spritelist.remove(sprite)