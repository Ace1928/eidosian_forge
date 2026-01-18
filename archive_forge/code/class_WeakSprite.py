from the Sprite and AbstractGroup classes below, it is strongly recommended
from weakref import WeakSet
from warnings import warn
import pygame
from pygame.rect import Rect
from pygame.time import get_ticks
from pygame.mask import from_surface
class WeakSprite(Sprite):
    """A subclass of Sprite that references its Groups weakly. This
    means that any group this belongs to that is not referenced anywhere
    else is garbage collected automatically.
    """

    def __init__(self, *groups):
        super().__init__(*groups)
        self.__dict__['_Sprite__g'] = WeakSet(self._Sprite__g)