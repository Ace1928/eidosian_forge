import os
import unittest
from pygame.tests import test_utils
from pygame.tests.test_utils import (
import pygame
from pygame.locals import *
from pygame.bufferproxy import BufferProxy
import platform
import gc
import weakref
import ctypes
class SurfaceFillTest(unittest.TestCase):

    def setUp(self):
        pygame.display.init()

    def tearDown(self):
        pygame.display.quit()

    def test_fill(self):
        screen = pygame.display.set_mode((640, 480))
        screen.fill((0, 255, 0), (0, 0, 320, 240))
        screen.fill((0, 255, 0), (320, 240, 320, 240))
        screen.fill((0, 0, 255), (320, 0, 320, 240))
        screen.fill((0, 0, 255), (0, 240, 320, 240))
        screen.set_clip((0, 0, 320, 480))
        screen.fill((255, 0, 0, 127), (160, 0, 320, 30), 0)
        screen.fill((255, 0, 0, 127), (160, 30, 320, 30), pygame.BLEND_ADD)
        screen.fill((0, 127, 127, 127), (160, 60, 320, 30), pygame.BLEND_SUB)
        screen.fill((0, 63, 63, 127), (160, 90, 320, 30), pygame.BLEND_MULT)
        screen.fill((0, 127, 127, 127), (160, 120, 320, 30), pygame.BLEND_MIN)
        screen.fill((127, 0, 0, 127), (160, 150, 320, 30), pygame.BLEND_MAX)
        screen.fill((255, 0, 0, 127), (160, 180, 320, 30), pygame.BLEND_RGBA_ADD)
        screen.fill((0, 127, 127, 127), (160, 210, 320, 30), pygame.BLEND_RGBA_SUB)
        screen.fill((0, 63, 63, 127), (160, 240, 320, 30), pygame.BLEND_RGBA_MULT)
        screen.fill((0, 127, 127, 127), (160, 270, 320, 30), pygame.BLEND_RGBA_MIN)
        screen.fill((127, 0, 0, 127), (160, 300, 320, 30), pygame.BLEND_RGBA_MAX)
        screen.fill((255, 0, 0, 127), (160, 330, 320, 30), pygame.BLEND_RGB_ADD)
        screen.fill((0, 127, 127, 127), (160, 360, 320, 30), pygame.BLEND_RGB_SUB)
        screen.fill((0, 63, 63, 127), (160, 390, 320, 30), pygame.BLEND_RGB_MULT)
        screen.fill((0, 127, 127, 127), (160, 420, 320, 30), pygame.BLEND_RGB_MIN)
        screen.fill((255, 0, 0, 127), (160, 450, 320, 30), pygame.BLEND_RGB_MAX)
        pygame.display.flip()
        for y in range(5, 480, 10):
            self.assertEqual(screen.get_at((10, y)), screen.get_at((330, 480 - y)))