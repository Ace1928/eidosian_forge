import unittest
import os
import sys
import time
import pygame, pygame.transform
from pygame.tests.test_utils import question
from pygame import display
class DisplayInteractiveTest(unittest.TestCase):
    __tags__ = ['interactive']

    def test_set_icon_interactive(self):
        os.environ['SDL_VIDEO_WINDOW_POS'] = '100,250'
        pygame.display.quit()
        pygame.display.init()
        test_icon = pygame.Surface((32, 32))
        test_icon.fill((255, 0, 0))
        pygame.display.set_icon(test_icon)
        screen = pygame.display.set_mode((400, 100))
        pygame.display.set_caption('Is the window icon a red square?')
        response = question('Is the display icon red square?')
        self.assertTrue(response)
        pygame.display.quit()

    def test_set_gamma_ramp(self):
        os.environ['SDL_VIDEO_WINDOW_POS'] = '100,250'
        pygame.display.quit()
        pygame.display.init()
        screen = pygame.display.set_mode((400, 100))
        screen.fill((100, 100, 100))
        blue_ramp = [x * 256 for x in range(0, 256)]
        blue_ramp[100] = 150 * 256
        normal_ramp = [x * 256 for x in range(0, 256)]
        gamma_success = False
        if pygame.display.set_gamma_ramp(normal_ramp, normal_ramp, blue_ramp):
            pygame.display.update()
            gamma_success = True
        if gamma_success:
            response = question('Is the window background tinted blue?')
            self.assertTrue(response)
            pygame.display.set_gamma_ramp(normal_ramp, normal_ramp, normal_ramp)
        pygame.display.quit()