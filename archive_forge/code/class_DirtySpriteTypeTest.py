import unittest
import pygame
from pygame import sprite
class DirtySpriteTypeTest(SpriteBase, unittest.TestCase):
    Sprite = sprite.DirtySprite
    Groups = [sprite.Group, sprite.LayeredUpdates, sprite.RenderUpdates, sprite.OrderedUpdates, sprite.LayeredDirty]