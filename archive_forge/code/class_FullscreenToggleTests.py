import unittest
import os
import sys
import time
import pygame, pygame.transform
from pygame.tests.test_utils import question
from pygame import display
class FullscreenToggleTests(unittest.TestCase):
    __tags__ = ['interactive']
    screen = None
    font = None
    isfullscreen = False
    WIDTH = 800
    HEIGHT = 600

    def setUp(self):
        pygame.init()
        if sys.platform == 'win32':
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT), flags=pygame.SCALED)
        else:
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption('Fullscreen Tests')
        self.screen.fill((255, 255, 255))
        pygame.display.flip()
        self.font = pygame.font.Font(None, 32)

    def tearDown(self):
        if self.isfullscreen:
            pygame.display.toggle_fullscreen()
        pygame.quit()

    def visual_test(self, fullscreen=False):
        text = ''
        if fullscreen:
            if not self.isfullscreen:
                pygame.display.toggle_fullscreen()
                self.isfullscreen = True
            text = 'Is this in fullscreen? [y/n]'
        else:
            if self.isfullscreen:
                pygame.display.toggle_fullscreen()
                self.isfullscreen = False
            text = 'Is this not in fullscreen [y/n]'
        s = self.font.render(text, False, (0, 0, 0))
        self.screen.blit(s, (self.WIDTH / 2 - self.font.size(text)[0] / 2, 100))
        pygame.display.flip()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return False
                    if event.key == pygame.K_y:
                        return True
                    if event.key == pygame.K_n:
                        return False
                if event.type == pygame.QUIT:
                    return False

    def test_fullscreen_true(self):
        self.assertTrue(self.visual_test(fullscreen=True))

    def test_fullscreen_false(self):
        self.assertTrue(self.visual_test(fullscreen=False))