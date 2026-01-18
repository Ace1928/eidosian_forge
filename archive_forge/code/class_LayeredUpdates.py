from the Sprite and AbstractGroup classes below, it is strongly recommended
from weakref import WeakSet
from warnings import warn
import pygame
from pygame.rect import Rect
from pygame.time import get_ticks
from pygame.mask import from_surface
class LayeredUpdates(AbstractGroup):
    """LayeredUpdates Group handles layers, which are drawn like OrderedUpdates

    pygame.sprite.LayeredUpdates(*sprites, **kwargs): return LayeredUpdates

    This group is fully compatible with pygame.sprite.Sprite.
    New in pygame 1.8.0

    """
    _init_rect = Rect(0, 0, 0, 0)

    def __init__(self, *sprites, **kwargs):
        """initialize an instance of LayeredUpdates with the given attributes

        You can set the default layer through kwargs using 'default_layer'
        and an integer for the layer. The default layer is 0.

        If the sprite you add has an attribute _layer, then that layer will be
        used. If **kwarg contains 'layer', then the passed sprites will be
        added to that layer (overriding the sprite._layer attribute). If
        neither the sprite nor **kwarg has a 'layer', then the default layer is
        used to add the sprites.

        """
        self._spritelayers = {}
        self._spritelist = []
        AbstractGroup.__init__(self)
        self._default_layer = kwargs.get('default_layer', 0)
        self.add(*sprites, **kwargs)

    def add_internal(self, sprite, layer=None):
        """Do not use this method directly.

        It is used by the group to add a sprite internally.

        """
        self.spritedict[sprite] = self._init_rect
        if layer is None:
            try:
                layer = sprite.layer
            except AttributeError:
                layer = self._default_layer
                setattr(sprite, '_layer', layer)
        elif hasattr(sprite, '_layer'):
            setattr(sprite, '_layer', layer)
        sprites = self._spritelist
        sprites_layers = self._spritelayers
        sprites_layers[sprite] = layer
        leng = len(sprites)
        low = mid = 0
        high = leng - 1
        while low <= high:
            mid = low + (high - low) // 2
            if sprites_layers[sprites[mid]] <= layer:
                low = mid + 1
            else:
                high = mid - 1
        while mid < leng and sprites_layers[sprites[mid]] <= layer:
            mid += 1
        sprites.insert(mid, sprite)

    def add(self, *sprites, **kwargs):
        """add a sprite or sequence of sprites to a group

        LayeredUpdates.add(*sprites, **kwargs): return None

        If the sprite you add has an attribute _layer, then that layer will be
        used. If **kwarg contains 'layer', then the passed sprites will be
        added to that layer (overriding the sprite._layer attribute). If
        neither the sprite nor **kwarg has a 'layer', then the default layer is
        used to add the sprites.

        """
        if not sprites:
            return
        layer = kwargs['layer'] if 'layer' in kwargs else None
        for sprite in sprites:
            if isinstance(sprite, Sprite):
                if not self.has_internal(sprite):
                    self.add_internal(sprite, layer)
                    sprite.add_internal(self)
            else:
                try:
                    self.add(*sprite, **kwargs)
                except (TypeError, AttributeError):
                    if hasattr(sprite, '_spritegroup'):
                        for spr in sprite.sprites():
                            if not self.has_internal(spr):
                                self.add_internal(spr, layer)
                                spr.add_internal(self)
                    elif not self.has_internal(sprite):
                        self.add_internal(sprite, layer)
                        sprite.add_internal(self)

    def remove_internal(self, sprite):
        """Do not use this method directly.

        The group uses it to add a sprite.

        """
        self._spritelist.remove(sprite)
        old_rect = self.spritedict[sprite]
        if old_rect is not self._init_rect:
            self.lostsprites.append(old_rect)
        if hasattr(sprite, 'rect'):
            self.lostsprites.append(sprite.rect)
        del self.spritedict[sprite]
        del self._spritelayers[sprite]

    def sprites(self):
        """return a ordered list of sprites (first back, last top).

        LayeredUpdates.sprites(): return sprites

        """
        return self._spritelist.copy()

    def draw(self, surface, bgsurf=None, special_flags=0):
        """draw all sprites in the right order onto the passed surface

        LayeredUpdates.draw(surface, special_flags=0): return Rect_list

        """
        spritedict = self.spritedict
        surface_blit = surface.blit
        dirty = self.lostsprites
        self.lostsprites = []
        dirty_append = dirty.append
        init_rect = self._init_rect
        for spr in self.sprites():
            rec = spritedict[spr]
            newrect = surface_blit(spr.image, spr.rect, None, special_flags)
            if rec is init_rect:
                dirty_append(newrect)
            elif newrect.colliderect(rec):
                dirty_append(newrect.union(rec))
            else:
                dirty_append(newrect)
                dirty_append(rec)
            spritedict[spr] = newrect
        return dirty

    def get_sprites_at(self, pos):
        """return a list with all sprites at that position

        LayeredUpdates.get_sprites_at(pos): return colliding_sprites

        Bottom sprites are listed first; the top ones are listed last.

        """
        _sprites = self._spritelist
        rect = Rect(pos, (1, 1))
        colliding_idx = rect.collidelistall(_sprites)
        return [_sprites[i] for i in colliding_idx]

    def get_sprite(self, idx):
        """return the sprite at the index idx from the groups sprites

        LayeredUpdates.get_sprite(idx): return sprite

        Raises IndexOutOfBounds if the idx is not within range.

        """
        return self._spritelist[idx]

    def remove_sprites_of_layer(self, layer_nr):
        """remove all sprites from a layer and return them as a list

        LayeredUpdates.remove_sprites_of_layer(layer_nr): return sprites

        """
        sprites = self.get_sprites_from_layer(layer_nr)
        self.remove(*sprites)
        return sprites

    def layers(self):
        """return a list of unique defined layers defined.

        LayeredUpdates.layers(): return layers

        """
        return sorted(set(self._spritelayers.values()))

    def change_layer(self, sprite, new_layer):
        """change the layer of the sprite

        LayeredUpdates.change_layer(sprite, new_layer): return None

        The sprite must have been added to the renderer already. This is not
        checked.

        """
        sprites = self._spritelist
        sprites_layers = self._spritelayers
        sprites.remove(sprite)
        sprites_layers.pop(sprite)
        leng = len(sprites)
        low = mid = 0
        high = leng - 1
        while low <= high:
            mid = low + (high - low) // 2
            if sprites_layers[sprites[mid]] <= new_layer:
                low = mid + 1
            else:
                high = mid - 1
        while mid < leng and sprites_layers[sprites[mid]] <= new_layer:
            mid += 1
        sprites.insert(mid, sprite)
        if hasattr(sprite, '_layer'):
            setattr(sprite, '_layer', new_layer)
        sprites_layers[sprite] = new_layer

    def get_layer_of_sprite(self, sprite):
        """return the layer that sprite is currently in

        If the sprite is not found, then it will return the default layer.

        """
        return self._spritelayers.get(sprite, self._default_layer)

    def get_top_layer(self):
        """return the top layer

        LayeredUpdates.get_top_layer(): return layer

        """
        return self._spritelayers[self._spritelist[-1]]

    def get_bottom_layer(self):
        """return the bottom layer

        LayeredUpdates.get_bottom_layer(): return layer

        """
        return self._spritelayers[self._spritelist[0]]

    def move_to_front(self, sprite):
        """bring the sprite to front layer

        LayeredUpdates.move_to_front(sprite): return None

        Brings the sprite to front by changing the sprite layer to the top-most
        layer. The sprite is added at the end of the list of sprites in that
        top-most layer.

        """
        self.change_layer(sprite, self.get_top_layer())

    def move_to_back(self, sprite):
        """move the sprite to the bottom layer

        LayeredUpdates.move_to_back(sprite): return None

        Moves the sprite to the bottom layer by moving it to a new layer below
        the current bottom layer.

        """
        self.change_layer(sprite, self.get_bottom_layer() - 1)

    def get_top_sprite(self):
        """return the topmost sprite

        LayeredUpdates.get_top_sprite(): return Sprite

        """
        return self._spritelist[-1]

    def get_sprites_from_layer(self, layer):
        """return all sprites from a layer ordered as they where added

        LayeredUpdates.get_sprites_from_layer(layer): return sprites

        Returns all sprites from a layer. The sprites are ordered in the
        sequence that they where added. (The sprites are not removed from the
        layer.

        """
        sprites = []
        sprites_append = sprites.append
        sprite_layers = self._spritelayers
        for spr in self._spritelist:
            if sprite_layers[spr] == layer:
                sprites_append(spr)
            elif sprite_layers[spr] > layer:
                break
        return sprites

    def switch_layer(self, layer1_nr, layer2_nr):
        """switch the sprites from layer1_nr to layer2_nr

        LayeredUpdates.switch_layer(layer1_nr, layer2_nr): return None

        The layers number must exist. This method does not check for the
        existence of the given layers.

        """
        sprites1 = self.remove_sprites_of_layer(layer1_nr)
        for spr in self.get_sprites_from_layer(layer2_nr):
            self.change_layer(spr, layer1_nr)
        self.add(*sprites1, layer=layer2_nr)