import unittest
import pygame
from pygame import sprite
class LayeredGroupBase:

    def test_get_layer_of_sprite(self):
        expected_layer = 666
        spr = self.sprite()
        self.LG.add(spr, layer=expected_layer)
        layer = self.LG.get_layer_of_sprite(spr)
        self.assertEqual(len(self.LG._spritelist), 1)
        self.assertEqual(layer, self.LG.get_layer_of_sprite(spr))
        self.assertEqual(layer, expected_layer)
        self.assertEqual(layer, self.LG._spritelayers[spr])

    def test_add(self):
        expected_layer = self.LG._default_layer
        spr = self.sprite()
        self.LG.add(spr)
        layer = self.LG.get_layer_of_sprite(spr)
        self.assertEqual(len(self.LG._spritelist), 1)
        self.assertEqual(layer, expected_layer)

    def test_add__sprite_with_layer_attribute(self):
        expected_layer = 100
        spr = self.sprite()
        spr._layer = expected_layer
        self.LG.add(spr)
        layer = self.LG.get_layer_of_sprite(spr)
        self.assertEqual(len(self.LG._spritelist), 1)
        self.assertEqual(layer, expected_layer)

    def test_add__passing_layer_keyword(self):
        expected_layer = 100
        spr = self.sprite()
        self.LG.add(spr, layer=expected_layer)
        layer = self.LG.get_layer_of_sprite(spr)
        self.assertEqual(len(self.LG._spritelist), 1)
        self.assertEqual(layer, expected_layer)

    def test_add__overriding_sprite_layer_attr(self):
        expected_layer = 200
        spr = self.sprite()
        spr._layer = 100
        self.LG.add(spr, layer=expected_layer)
        layer = self.LG.get_layer_of_sprite(spr)
        self.assertEqual(len(self.LG._spritelist), 1)
        self.assertEqual(layer, expected_layer)

    def test_add__adding_sprite_on_init(self):
        spr = self.sprite()
        lrg2 = sprite.LayeredUpdates(spr)
        expected_layer = lrg2._default_layer
        layer = lrg2._spritelayers[spr]
        self.assertEqual(len(lrg2._spritelist), 1)
        self.assertEqual(layer, expected_layer)

    def test_add__sprite_init_layer_attr(self):
        expected_layer = 20
        spr = self.sprite()
        spr._layer = expected_layer
        lrg2 = sprite.LayeredUpdates(spr)
        layer = lrg2._spritelayers[spr]
        self.assertEqual(len(lrg2._spritelist), 1)
        self.assertEqual(layer, expected_layer)

    def test_add__sprite_init_passing_layer(self):
        expected_layer = 33
        spr = self.sprite()
        lrg2 = sprite.LayeredUpdates(spr, layer=expected_layer)
        layer = lrg2._spritelayers[spr]
        self.assertEqual(len(lrg2._spritelist), 1)
        self.assertEqual(layer, expected_layer)

    def test_add__sprite_init_overiding_layer(self):
        expected_layer = 33
        spr = self.sprite()
        spr._layer = 55
        lrg2 = sprite.LayeredUpdates(spr, layer=expected_layer)
        layer = lrg2._spritelayers[spr]
        self.assertEqual(len(lrg2._spritelist), 1)
        self.assertEqual(layer, expected_layer)

    def test_add__spritelist(self):
        expected_layer = self.LG._default_layer
        sprite_count = 10
        sprites = [self.sprite() for _ in range(sprite_count)]
        self.LG.add(sprites)
        self.assertEqual(len(self.LG._spritelist), sprite_count)
        for i in range(sprite_count):
            layer = self.LG.get_layer_of_sprite(sprites[i])
            self.assertEqual(layer, expected_layer)

    def test_add__spritelist_with_layer_attr(self):
        sprites = []
        sprite_and_layer_count = 10
        for i in range(sprite_and_layer_count):
            sprites.append(self.sprite())
            sprites[-1]._layer = i
        self.LG.add(sprites)
        self.assertEqual(len(self.LG._spritelist), sprite_and_layer_count)
        for i in range(sprite_and_layer_count):
            layer = self.LG.get_layer_of_sprite(sprites[i])
            self.assertEqual(layer, i)

    def test_add__spritelist_passing_layer(self):
        expected_layer = 33
        sprite_count = 10
        sprites = [self.sprite() for _ in range(sprite_count)]
        self.LG.add(sprites, layer=expected_layer)
        self.assertEqual(len(self.LG._spritelist), sprite_count)
        for i in range(sprite_count):
            layer = self.LG.get_layer_of_sprite(sprites[i])
            self.assertEqual(layer, expected_layer)

    def test_add__spritelist_overriding_layer(self):
        expected_layer = 33
        sprites = []
        sprite_and_layer_count = 10
        for i in range(sprite_and_layer_count):
            sprites.append(self.sprite())
            sprites[-1].layer = i
        self.LG.add(sprites, layer=expected_layer)
        self.assertEqual(len(self.LG._spritelist), sprite_and_layer_count)
        for i in range(sprite_and_layer_count):
            layer = self.LG.get_layer_of_sprite(sprites[i])
            self.assertEqual(layer, expected_layer)

    def test_add__spritelist_init(self):
        sprite_count = 10
        sprites = [self.sprite() for _ in range(sprite_count)]
        lrg2 = sprite.LayeredUpdates(sprites)
        expected_layer = lrg2._default_layer
        self.assertEqual(len(lrg2._spritelist), sprite_count)
        for i in range(sprite_count):
            layer = lrg2.get_layer_of_sprite(sprites[i])
            self.assertEqual(layer, expected_layer)

    def test_remove__sprite(self):
        sprites = []
        sprite_count = 10
        for i in range(sprite_count):
            sprites.append(self.sprite())
            sprites[-1].rect = pygame.Rect((0, 0, 0, 0))
        self.LG.add(sprites)
        self.assertEqual(len(self.LG._spritelist), sprite_count)
        for i in range(sprite_count):
            self.LG.remove(sprites[i])
        self.assertEqual(len(self.LG._spritelist), 0)

    def test_sprites(self):
        sprites = []
        sprite_and_layer_count = 10
        for i in range(sprite_and_layer_count, 0, -1):
            sprites.append(self.sprite())
            sprites[-1]._layer = i
        self.LG.add(sprites)
        self.assertEqual(len(self.LG._spritelist), sprite_and_layer_count)
        expected_sprites = list(reversed(sprites))
        actual_sprites = self.LG.sprites()
        self.assertListEqual(actual_sprites, expected_sprites)

    def test_layers(self):
        sprites = []
        expected_layers = []
        layer_count = 10
        for i in range(layer_count):
            expected_layers.append(i)
            for j in range(5):
                sprites.append(self.sprite())
                sprites[-1]._layer = i
        self.LG.add(sprites)
        layers = self.LG.layers()
        self.assertListEqual(layers, expected_layers)

    def test_add__layers_are_correct(self):
        layers = [1, 4, 6, 8, 3, 6, 2, 6, 4, 5, 6, 1, 0, 9, 7, 6, 54, 8, 2, 43, 6, 1]
        for lay in layers:
            self.LG.add(self.sprite(), layer=lay)
        layers.sort()
        for idx, spr in enumerate(self.LG.sprites()):
            layer = self.LG.get_layer_of_sprite(spr)
            self.assertEqual(layer, layers[idx])

    def test_change_layer(self):
        expected_layer = 99
        spr = self.sprite()
        self.LG.add(spr, layer=expected_layer)
        self.assertEqual(self.LG._spritelayers[spr], expected_layer)
        expected_layer = 44
        self.LG.change_layer(spr, expected_layer)
        self.assertEqual(self.LG._spritelayers[spr], expected_layer)
        expected_layer = 77
        spr2 = self.sprite()
        spr2.layer = 55
        self.LG.add(spr2)
        self.LG.change_layer(spr2, expected_layer)
        self.assertEqual(spr2.layer, expected_layer)

    def test_get_sprites_at(self):
        sprites = []
        expected_sprites = []
        for i in range(3):
            spr = self.sprite()
            spr.rect = pygame.Rect(i * 50, i * 50, 100, 100)
            sprites.append(spr)
            if i < 2:
                expected_sprites.append(spr)
        self.LG.add(sprites)
        result = self.LG.get_sprites_at((50, 50))
        self.assertEqual(result, expected_sprites)

    def test_get_top_layer(self):
        layers = [1, 5, 2, 8, 4, 5, 3, 88, 23, 0]
        for i in layers:
            self.LG.add(self.sprite(), layer=i)
        top_layer = self.LG.get_top_layer()
        self.assertEqual(top_layer, self.LG.get_top_layer())
        self.assertEqual(top_layer, max(layers))
        self.assertEqual(top_layer, max(self.LG._spritelayers.values()))
        self.assertEqual(top_layer, self.LG._spritelayers[self.LG._spritelist[-1]])

    def test_get_bottom_layer(self):
        layers = [1, 5, 2, 8, 4, 5, 3, 88, 23, 0]
        for i in layers:
            self.LG.add(self.sprite(), layer=i)
        bottom_layer = self.LG.get_bottom_layer()
        self.assertEqual(bottom_layer, self.LG.get_bottom_layer())
        self.assertEqual(bottom_layer, min(layers))
        self.assertEqual(bottom_layer, min(self.LG._spritelayers.values()))
        self.assertEqual(bottom_layer, self.LG._spritelayers[self.LG._spritelist[0]])

    def test_move_to_front(self):
        layers = [1, 5, 2, 8, 4, 5, 3, 88, 23, 0]
        for i in layers:
            self.LG.add(self.sprite(), layer=i)
        spr = self.sprite()
        self.LG.add(spr, layer=3)
        self.assertNotEqual(spr, self.LG._spritelist[-1])
        self.LG.move_to_front(spr)
        self.assertEqual(spr, self.LG._spritelist[-1])

    def test_move_to_back(self):
        layers = [1, 5, 2, 8, 4, 5, 3, 88, 23, 0]
        for i in layers:
            self.LG.add(self.sprite(), layer=i)
        spr = self.sprite()
        self.LG.add(spr, layer=55)
        self.assertNotEqual(spr, self.LG._spritelist[0])
        self.LG.move_to_back(spr)
        self.assertEqual(spr, self.LG._spritelist[0])

    def test_get_top_sprite(self):
        layers = [1, 5, 2, 8, 4, 5, 3, 88, 23, 0]
        for i in layers:
            self.LG.add(self.sprite(), layer=i)
        expected_layer = self.LG.get_top_layer()
        layer = self.LG.get_layer_of_sprite(self.LG.get_top_sprite())
        self.assertEqual(layer, expected_layer)

    def test_get_sprites_from_layer(self):
        sprites = {}
        layers = [1, 4, 5, 6, 3, 7, 8, 2, 1, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 6, 5, 4, 3, 2]
        for lay in layers:
            spr = self.sprite()
            spr._layer = lay
            self.LG.add(spr)
            if lay not in sprites:
                sprites[lay] = []
            sprites[lay].append(spr)
        for lay in self.LG.layers():
            for spr in self.LG.get_sprites_from_layer(lay):
                self.assertIn(spr, sprites[lay])
                sprites[lay].remove(spr)
                if len(sprites[lay]) == 0:
                    del sprites[lay]
        self.assertEqual(len(sprites.values()), 0)

    def test_switch_layer(self):
        sprites1 = []
        sprites2 = []
        layers = [3, 2, 3, 2, 3, 3, 2, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 3, 2, 2, 3, 2, 3]
        for lay in layers:
            spr = self.sprite()
            spr._layer = lay
            self.LG.add(spr)
            if lay == 2:
                sprites1.append(spr)
            else:
                sprites2.append(spr)
        sprites1.sort(key=id)
        sprites2.sort(key=id)
        layer2_sprites = sorted(self.LG.get_sprites_from_layer(2), key=id)
        layer3_sprites = sorted(self.LG.get_sprites_from_layer(3), key=id)
        self.assertListEqual(sprites1, layer2_sprites)
        self.assertListEqual(sprites2, layer3_sprites)
        self.assertEqual(len(self.LG), len(sprites1) + len(sprites2))
        self.LG.switch_layer(2, 3)
        layer2_sprites = sorted(self.LG.get_sprites_from_layer(2), key=id)
        layer3_sprites = sorted(self.LG.get_sprites_from_layer(3), key=id)
        self.assertListEqual(sprites1, layer3_sprites)
        self.assertListEqual(sprites2, layer2_sprites)
        self.assertEqual(len(self.LG), len(sprites1) + len(sprites2))

    def test_copy(self):
        self.LG.add(self.sprite())
        spr = self.LG.sprites()[0]
        lg_copy = self.LG.copy()
        self.assertIsInstance(lg_copy, type(self.LG))
        self.assertIn(spr, lg_copy)
        self.assertIn(lg_copy, spr.groups())