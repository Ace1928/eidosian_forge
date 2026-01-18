from the Sprite and AbstractGroup classes below, it is strongly recommended
from weakref import WeakSet
from warnings import warn
import pygame
from pygame.rect import Rect
from pygame.time import get_ticks
from pygame.mask import from_surface
class DirtySprite(Sprite):
    """a more featureful subclass of Sprite with more attributes

    pygame.sprite.DirtySprite(*groups): return DirtySprite

    Extra DirtySprite attributes with their default values:

    dirty = 1
        If set to 1, it is repainted and then set to 0 again.
        If set to 2, it is always dirty (repainted each frame;
        flag is not reset).
        If set to 0, it is not dirty and therefore not repainted again.

    blendmode = 0
        It's the special_flags argument of Surface.blit; see the blendmodes in
        the Surface.blit documentation

    source_rect = None
        This is the source rect to use. Remember that it is relative to the top
        left corner (0, 0) of self.image.

    visible = 1
        Normally this is 1. If set to 0, it will not be repainted. (If you
        change visible to 1, you must set dirty to 1 for it to be erased from
        the screen.)

    _layer = 0
        0 is the default value but this is able to be set differently
        when subclassing.

    """

    def __init__(self, *groups):
        self.dirty = 1
        self.blendmode = 0
        self._visible = 1
        self._layer = getattr(self, '_layer', 0)
        self.source_rect = None
        Sprite.__init__(self, *groups)

    def _set_visible(self, val):
        """set the visible value (0 or 1) and makes the sprite dirty"""
        self._visible = val
        if self.dirty < 2:
            self.dirty = 1

    def _get_visible(self):
        """return the visible value of that sprite"""
        return self._visible

    @property
    def visible(self):
        """
        You can make this sprite disappear without removing it from the group
        assign 0 for invisible and 1 for visible
        """
        return self._get_visible()

    @visible.setter
    def visible(self, value):
        self._set_visible(value)

    @property
    def layer(self):
        """
        Layer property can only be set before the sprite is added to a group,
        after that it is read only and a sprite's layer in a group should be
        set via the group's change_layer() method.

        Overwrites dynamic property from sprite class for speed.
        """
        return self._layer

    @layer.setter
    def layer(self, value):
        if not self.alive():
            self._layer = value
        else:
            raise AttributeError("Can't set layer directly after adding to group. Use group.change_layer(sprite, new_layer) instead.")

    def __repr__(self):
        return f'<{self.__class__.__name__} DirtySprite(in {len(self.groups())} groups)>'