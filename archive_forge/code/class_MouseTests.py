import unittest
import os
import platform
import warnings
import pygame
class MouseTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pygame.display.init()

    @classmethod
    def tearDownClass(cls):
        pygame.display.quit()