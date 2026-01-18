from re import T
import sys
import os
import unittest
import pathlib
import platform
import pygame
from pygame import font as pygame_font  # So font can be replaced with ftfont
@unittest.skipIf(IS_PYPY, 'pypy skip known failure')
class FontTest(unittest.TestCase):

    def setUp(self):
        pygame_font.init()

    def tearDown(self):
        pygame_font.quit()

    def test_render_args(self):
        screen = pygame.display.set_mode((600, 400))
        rect = screen.get_rect()
        f = pygame_font.Font(None, 20)
        screen.fill((10, 10, 10))
        font_surface = f.render('   bar', True, (0, 0, 0), (255, 255, 255))
        font_rect = font_surface.get_rect()
        font_rect.topleft = rect.topleft
        self.assertTrue(font_surface)
        screen.blit(font_surface, font_rect, font_rect)
        pygame.display.update()
        self.assertEqual(tuple(screen.get_at((0, 0)))[:3], (255, 255, 255))
        self.assertEqual(tuple(screen.get_at(font_rect.topleft))[:3], (255, 255, 255))
        if os.environ.get('SDL_VIDEODRIVER') != 'dummy':
            screen.fill((10, 10, 10))
            font_surface = f.render('   bar', True, (0, 0, 0), None)
            font_rect = font_surface.get_rect()
            font_rect.topleft = rect.topleft
            self.assertTrue(font_surface)
            screen.blit(font_surface, font_rect, font_rect)
            pygame.display.update()
            self.assertEqual(tuple(screen.get_at((0, 0)))[:3], (10, 10, 10))
            self.assertEqual(tuple(screen.get_at(font_rect.topleft))[:3], (10, 10, 10))
            screen.fill((10, 10, 10))
            font_surface = f.render('   bar', True, (0, 0, 0))
            font_rect = font_surface.get_rect()
            font_rect.topleft = rect.topleft
            self.assertTrue(font_surface)
            screen.blit(font_surface, font_rect, font_rect)
            pygame.display.update(rect)
            self.assertEqual(tuple(screen.get_at((0, 0)))[:3], (10, 10, 10))
            self.assertEqual(tuple(screen.get_at(font_rect.topleft))[:3], (10, 10, 10))