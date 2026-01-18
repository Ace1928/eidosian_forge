from the Sprite and AbstractGroup classes below, it is strongly recommended
from weakref import WeakSet
from warnings import warn
import pygame
from pygame.rect import Rect
from pygame.time import get_ticks
from pygame.mask import from_surface
class RenderUpdates(Group):
    """Group class that tracks dirty updates

    pygame.sprite.RenderUpdates(*sprites): return RenderUpdates

    This class is derived from pygame.sprite.Group(). It has an enhanced draw
    method that tracks the changed areas of the screen.

    """

    def draw(self, surface, bgsurf=None, special_flags=0):
        surface_blit = surface.blit
        dirty = self.lostsprites
        self.lostsprites = []
        dirty_append = dirty.append
        for sprite in self.sprites():
            old_rect = self.spritedict[sprite]
            new_rect = surface_blit(sprite.image, sprite.rect, None, special_flags)
            if old_rect:
                if new_rect.colliderect(old_rect):
                    dirty_append(new_rect.union(old_rect))
                else:
                    dirty_append(new_rect)
                    dirty_append(old_rect)
            else:
                dirty_append(new_rect)
            self.spritedict[sprite] = new_rect
        return dirty