from the Sprite and AbstractGroup classes below, it is strongly recommended
from weakref import WeakSet
from warnings import warn
import pygame
from pygame.rect import Rect
from pygame.time import get_ticks
from pygame.mask import from_surface
@staticmethod
def _find_dirty_area(_clip, _old_rect, _rect, _sprites, _update, _update_append, init_rect):
    for spr in _sprites:
        if spr.dirty > 0:
            if spr.source_rect:
                _union_rect = _rect(spr.rect.topleft, spr.source_rect.size)
            else:
                _union_rect = _rect(spr.rect)
            _union_rect_collidelist = _union_rect.collidelist
            _union_rect_union_ip = _union_rect.union_ip
            i = _union_rect_collidelist(_update)
            while i > -1:
                _union_rect_union_ip(_update[i])
                del _update[i]
                i = _union_rect_collidelist(_update)
            _update_append(_union_rect.clip(_clip))
            if _old_rect[spr] is not init_rect:
                _union_rect = _rect(_old_rect[spr])
                _union_rect_collidelist = _union_rect.collidelist
                _union_rect_union_ip = _union_rect.union_ip
                i = _union_rect_collidelist(_update)
                while i > -1:
                    _union_rect_union_ip(_update[i])
                    del _update[i]
                    i = _union_rect_collidelist(_update)
                _update_append(_union_rect.clip(_clip))