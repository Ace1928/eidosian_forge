import unittest
import pygame
from pygame import sprite
class SingleGroupBugsTest(unittest.TestCase):

    def test_memoryleak_bug(self):
        import weakref
        import gc

        class MySprite(sprite.Sprite):

            def __init__(self, *args, **kwargs):
                sprite.Sprite.__init__(self, *args, **kwargs)
                self.image = pygame.Surface((2, 4), 0, 24)
                self.rect = self.image.get_rect()
        g = sprite.GroupSingle()
        screen = pygame.Surface((4, 8), 0, 24)
        s = MySprite()
        r = weakref.ref(s)
        g.sprite = s
        del s
        gc.collect()
        self.assertIsNotNone(r())
        g.update()
        g.draw(screen)
        g.sprite = MySprite()
        gc.collect()
        self.assertIsNone(r())