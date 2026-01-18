import unittest
import pygame
from pygame import sprite
class SpriteBase:

    def setUp(self):
        self.groups = []
        for Group in self.Groups:
            self.groups.append(Group())
        self.sprite = self.Sprite()

    def test_add_internal(self):
        for g in self.groups:
            self.sprite.add_internal(g)
        for g in self.groups:
            self.assertIn(g, self.sprite.groups())

    def test_remove_internal(self):
        for g in self.groups:
            self.sprite.add_internal(g)
        for g in self.groups:
            self.sprite.remove_internal(g)
        for g in self.groups:
            self.assertFalse(g in self.sprite.groups())

    def test_update(self):

        class test_sprite(pygame.sprite.Sprite):
            sink = []

            def __init__(self, *groups):
                pygame.sprite.Sprite.__init__(self, *groups)

            def update(self, *args):
                self.sink += args
        s = test_sprite()
        s.update(1, 2, 3)
        self.assertEqual(test_sprite.sink, [1, 2, 3])

    def test_update_with_kwargs(self):

        class test_sprite(pygame.sprite.Sprite):
            sink = []
            sink_dict = {}

            def __init__(self, *groups):
                pygame.sprite.Sprite.__init__(self, *groups)

            def update(self, *args, **kwargs):
                self.sink += args
                self.sink_dict.update(kwargs)
        s = test_sprite()
        s.update(1, 2, 3, foo=4, bar=5)
        self.assertEqual(test_sprite.sink, [1, 2, 3])
        self.assertEqual(test_sprite.sink_dict, {'foo': 4, 'bar': 5})

    def test___init____added_to_groups_passed(self):
        expected_groups = sorted(self.groups, key=id)
        sprite = self.Sprite(self.groups)
        groups = sorted(sprite.groups(), key=id)
        self.assertListEqual(groups, expected_groups)

    def test_add(self):
        expected_groups = sorted(self.groups, key=id)
        self.sprite.add(self.groups)
        groups = sorted(self.sprite.groups(), key=id)
        self.assertListEqual(groups, expected_groups)

    def test_alive(self):
        self.assertFalse(self.sprite.alive(), 'Sprite should not be alive if in no groups')
        self.sprite.add(self.groups)
        self.assertTrue(self.sprite.alive())

    def test_groups(self):
        for i, g in enumerate(self.groups):
            expected_groups = sorted(self.groups[:i + 1], key=id)
            self.sprite.add(g)
            groups = sorted(self.sprite.groups(), key=id)
            self.assertListEqual(groups, expected_groups)

    def test_kill(self):
        self.sprite.add(self.groups)
        self.assertTrue(self.sprite.alive())
        self.sprite.kill()
        self.assertListEqual(self.sprite.groups(), [])
        self.assertFalse(self.sprite.alive())

    def test_remove(self):
        self.sprite.add(self.groups)
        self.sprite.remove(self.groups)
        self.assertListEqual(self.sprite.groups(), [])