import unittest
import os
import platform
from pygame.tests import test_utils
from pygame.tests.test_utils import example_path
import pygame
import pygame.transform
from pygame.locals import *
class TransformDisplayModuleTest(unittest.TestCase):

    def setUp(self):
        pygame.display.init()
        pygame.display.set_mode((320, 200))

    def tearDown(self):
        pygame.display.quit()

    def test_flip(self):
        """honors the set_color key on the returned surface from flip."""
        image_loaded = pygame.image.load(example_path('data/chimp.png'))
        image = pygame.Surface(image_loaded.get_size(), 0, 32)
        image.blit(image_loaded, (0, 0))
        image_converted = image_loaded.convert()
        self.assertFalse(image.get_flags() & pygame.SRCALPHA)
        self.assertFalse(image_converted.get_flags() & pygame.SRCALPHA)
        surf = pygame.Surface(image.get_size(), 0, 32)
        surf2 = pygame.Surface(image.get_size(), 0, 32)
        surf.fill((255, 255, 255))
        surf2.fill((255, 255, 255))
        colorkey = image.get_at((0, 0))
        image.set_colorkey(colorkey, RLEACCEL)
        timage = pygame.transform.flip(image, 1, 0)
        colorkey = image_converted.get_at((0, 0))
        image_converted.set_colorkey(colorkey, RLEACCEL)
        timage_converted = pygame.transform.flip(surface=image_converted, flip_x=1, flip_y=0)
        surf.blit(timage, (0, 0))
        surf2.blit(image, (0, 0))
        self.assertEqual(surf.get_at((0, 0)), surf2.get_at((0, 0)))
        self.assertEqual(surf2.get_at((0, 0)), (255, 255, 255, 255))
        surf.fill((255, 255, 255))
        surf2.fill((255, 255, 255))
        surf.blit(timage_converted, (0, 0))
        surf2.blit(image_converted, (0, 0))
        self.assertEqual(surf.get_at((0, 0)), surf2.get_at((0, 0)))

    def test_flip_alpha(self):
        """returns a surface with the same properties as the input."""
        image_loaded = pygame.image.load(example_path('data/chimp.png'))
        image_alpha = pygame.Surface(image_loaded.get_size(), pygame.SRCALPHA, 32)
        image_alpha.blit(image_loaded, (0, 0))
        surf = pygame.Surface(image_loaded.get_size(), 0, 32)
        surf2 = pygame.Surface(image_loaded.get_size(), 0, 32)
        colorkey = image_alpha.get_at((0, 0))
        image_alpha.set_colorkey(colorkey, RLEACCEL)
        timage_alpha = pygame.transform.flip(image_alpha, 1, 0)
        self.assertTrue(image_alpha.get_flags() & pygame.SRCALPHA)
        self.assertTrue(timage_alpha.get_flags() & pygame.SRCALPHA)
        surf.fill((255, 255, 255))
        surf2.fill((255, 255, 255))
        surf.blit(timage_alpha, (0, 0))
        surf2.blit(image_alpha, (0, 0))
        self.assertEqual(surf.get_at((0, 0)), surf2.get_at((0, 0)))
        self.assertEqual(surf2.get_at((0, 0)), (255, 0, 0, 255))