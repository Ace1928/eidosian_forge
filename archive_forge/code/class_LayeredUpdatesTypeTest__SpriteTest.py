import unittest
import pygame
from pygame import sprite
class LayeredUpdatesTypeTest__SpriteTest(LayeredGroupBase, unittest.TestCase):
    sprite = sprite.Sprite

    def setUp(self):
        self.LG = sprite.LayeredUpdates()