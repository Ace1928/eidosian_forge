from the Sprite and AbstractGroup classes below, it is strongly recommended
from weakref import WeakSet
from warnings import warn
import pygame
from pygame.rect import Rect
from pygame.time import get_ticks
from pygame.mask import from_surface
class GroupSingle(AbstractGroup):
    """A group container that holds a single most recent item.

    This class works just like a regular group, but it only keeps a single
    sprite in the group. Whatever sprite has been added to the group last will
    be the only sprite in the group.

    You can access its one sprite as the .sprite attribute.  Assigning to this
    attribute will properly remove the old sprite and then add the new one.

    """

    def __init__(self, sprite=None):
        AbstractGroup.__init__(self)
        self.__sprite = None
        if sprite is not None:
            self.add(sprite)

    def copy(self):
        return GroupSingle(self.__sprite)

    def sprites(self):
        if self.__sprite is not None:
            return [self.__sprite]
        return []

    def add_internal(self, sprite, layer=None):
        if self.__sprite is not None:
            self.__sprite.remove_internal(self)
            self.remove_internal(self.__sprite)
        self.__sprite = sprite

    def __bool__(self):
        return self.__sprite is not None

    def _get_sprite(self):
        return self.__sprite

    def _set_sprite(self, sprite):
        self.add_internal(sprite)
        sprite.add_internal(self)
        return sprite

    @property
    def sprite(self):
        """
        Property for the single sprite contained in this group

        :return: The sprite.
        """
        return self._get_sprite()

    @sprite.setter
    def sprite(self, sprite_to_set):
        self._set_sprite(sprite_to_set)

    def remove_internal(self, sprite):
        if sprite is self.__sprite:
            self.__sprite = None
        if sprite in self.spritedict:
            AbstractGroup.remove_internal(self, sprite)

    def has_internal(self, sprite):
        return self.__sprite is sprite

    def __contains__(self, sprite):
        return self.__sprite is sprite