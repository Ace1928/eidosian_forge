from re import T
import sys
import os
import unittest
import pathlib
import platform
import pygame
from pygame import font as pygame_font  # So font can be replaced with ftfont
@unittest.skipIf(IS_PYPY, 'pypy skip known failure')
class VisualTests(unittest.TestCase):
    __tags__ = ['interactive']
    screen = None
    aborted = False

    def setUp(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((600, 200))
            self.screen.fill((255, 255, 255))
            pygame.display.flip()
            self.f = pygame_font.Font(None, 32)

    def abort(self):
        if self.screen is not None:
            pygame.quit()
        self.aborted = True

    def query(self, bold=False, italic=False, underline=False, strikethrough=False, antialiase=False):
        if self.aborted:
            return False
        spacing = 10
        offset = 20
        y = spacing
        f = self.f
        screen = self.screen
        screen.fill((255, 255, 255))
        pygame.display.flip()
        if not (bold or italic or underline or strikethrough or antialiase):
            text = 'normal'
        else:
            modes = []
            if bold:
                modes.append('bold')
            if italic:
                modes.append('italic')
            if underline:
                modes.append('underlined')
            if strikethrough:
                modes.append('strikethrough')
            if antialiase:
                modes.append('antialiased')
            text = f'{'-'.join(modes)} (y/n):'
        f.set_bold(bold)
        f.set_italic(italic)
        f.set_underline(underline)
        if pygame_font.__name__ != 'pygame.ftfont':
            f.set_strikethrough(strikethrough)
        s = f.render(text, antialiase, (0, 0, 0))
        screen.blit(s, (offset, y))
        y += s.get_size()[1] + spacing
        f.set_bold(False)
        f.set_italic(False)
        f.set_underline(False)
        if pygame_font.__name__ != 'pygame.ftfont':
            f.set_strikethrough(False)
        s = f.render('(some comparison text)', False, (0, 0, 0))
        screen.blit(s, (offset, y))
        pygame.display.flip()
        while True:
            for evt in pygame.event.get():
                if evt.type == pygame.KEYDOWN:
                    if evt.key == pygame.K_ESCAPE:
                        self.abort()
                        return False
                    if evt.key == pygame.K_y:
                        return True
                    if evt.key == pygame.K_n:
                        return False
                if evt.type == pygame.QUIT:
                    self.abort()
                    return False

    def test_bold(self):
        self.assertTrue(self.query(bold=True))

    def test_italic(self):
        self.assertTrue(self.query(italic=True))

    def test_underline(self):
        self.assertTrue(self.query(underline=True))

    def test_strikethrough(self):
        if pygame_font.__name__ != 'pygame.ftfont':
            self.assertTrue(self.query(strikethrough=True))

    def test_antialiase(self):
        self.assertTrue(self.query(antialiase=True))

    def test_bold_antialiase(self):
        self.assertTrue(self.query(bold=True, antialiase=True))

    def test_italic_underline(self):
        self.assertTrue(self.query(italic=True, underline=True))

    def test_bold_strikethrough(self):
        if pygame_font.__name__ != 'pygame.ftfont':
            self.assertTrue(self.query(bold=True, strikethrough=True))