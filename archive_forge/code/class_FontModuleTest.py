from re import T
import sys
import os
import unittest
import pathlib
import platform
import pygame
from pygame import font as pygame_font  # So font can be replaced with ftfont
@unittest.skipIf(IS_PYPY, 'pypy skip known failure')
class FontModuleTest(unittest.TestCase):

    def setUp(self):
        pygame_font.init()

    def tearDown(self):
        pygame_font.quit()

    def test_get_sdl_ttf_version(self):

        def test_ver_tuple(ver):
            self.assertIsInstance(ver, tuple)
            self.assertEqual(len(ver), 3)
            for i in ver:
                self.assertIsInstance(i, int)
        if pygame_font.__name__ != 'pygame.ftfont':
            compiled = pygame_font.get_sdl_ttf_version()
            linked = pygame_font.get_sdl_ttf_version(linked=True)
            test_ver_tuple(compiled)
            test_ver_tuple(linked)
            self.assertTrue(linked >= compiled)

    def test_SysFont(self):
        fonts = pygame_font.get_fonts()
        if 'arial' in fonts:
            font_name = 'arial'
        else:
            font_name = sorted(fonts)[0]
        o = pygame_font.SysFont(font_name, 20)
        self.assertTrue(isinstance(o, pygame_font.FontType))
        o = pygame_font.SysFont(font_name, 20, italic=True)
        self.assertTrue(isinstance(o, pygame_font.FontType))
        o = pygame_font.SysFont(font_name, 20, bold=True)
        self.assertTrue(isinstance(o, pygame_font.FontType))
        o = pygame_font.SysFont('thisisnotafont', 20)
        self.assertTrue(isinstance(o, pygame_font.FontType))

    def test_get_default_font(self):
        self.assertEqual(pygame_font.get_default_font(), 'freesansbold.ttf')

    def test_get_fonts_returns_something(self):
        fnts = pygame_font.get_fonts()
        self.assertTrue(fnts)

    def test_get_fonts(self):
        fnts = pygame_font.get_fonts()
        self.assertTrue(fnts, msg=repr(fnts))
        for name in fnts:
            self.assertTrue(isinstance(name, str), name)
            self.assertFalse(any((c.isupper() for c in name)))
            self.assertTrue(name.isalnum(), name)

    def test_get_init(self):
        self.assertTrue(pygame_font.get_init())
        pygame_font.quit()
        self.assertFalse(pygame_font.get_init())

    def test_init(self):
        pygame_font.init()

    def test_match_font_all_exist(self):
        fonts = pygame_font.get_fonts()
        for font in fonts:
            path = pygame_font.match_font(font)
            self.assertFalse(path is None)
            self.assertTrue(os.path.isabs(path) and os.path.isfile(path))

    def test_match_font_name(self):
        """That match_font accepts names of various types"""
        font = pygame_font.get_fonts()[0]
        font_path = pygame_font.match_font(font)
        self.assertIsNotNone(font_path)
        font_b = font.encode()
        not_a_font = 'thisisnotafont'
        not_a_font_b = b'thisisnotafont'
        good_font_names = [font_b, ','.join([not_a_font, font, not_a_font]), [not_a_font, font, not_a_font], (name for name in [not_a_font, font, not_a_font]), b','.join([not_a_font_b, font_b, not_a_font_b]), [not_a_font_b, font_b, not_a_font_b], [font, not_a_font, font_b, not_a_font_b]]
        for font_name in good_font_names:
            self.assertEqual(pygame_font.match_font(font_name), font_path, font_name)

    def test_not_match_font_name(self):
        """match_font return None when names of various types do not exist"""
        not_a_font = 'thisisnotafont'
        not_a_font_b = b'thisisnotafont'
        bad_font_names = [not_a_font, ','.join([not_a_font, not_a_font, not_a_font]), [not_a_font, not_a_font, not_a_font], (name for name in [not_a_font, not_a_font, not_a_font]), not_a_font_b, b','.join([not_a_font_b, not_a_font_b, not_a_font_b]), [not_a_font_b, not_a_font_b, not_a_font_b], [not_a_font, not_a_font_b, not_a_font]]
        for font_name in bad_font_names:
            self.assertIsNone(pygame_font.match_font(font_name), font_name)

    def test_match_font_bold(self):
        fonts = pygame_font.get_fonts()
        self.assertTrue(any((pygame_font.match_font(font, bold=True) for font in fonts)))

    def test_match_font_italic(self):
        fonts = pygame_font.get_fonts()
        self.assertTrue(any((pygame_font.match_font(font, italic=True) for font in fonts)))

    def test_issue_742(self):
        """that the font background does not crash."""
        surf = pygame.Surface((320, 240))
        font = pygame_font.Font(None, 24)
        image = font.render('Test', 0, (255, 255, 255), (0, 0, 0))
        self.assertIsNone(image.get_colorkey())
        image.set_alpha(255)
        surf.blit(image, (0, 0))
        self.assertEqual(surf.get_at((0, 0)), pygame.Color(0, 0, 0))

    def test_issue_font_alphablit(self):
        """Check that blitting anti-aliased text doesn't
        change the background blue"""
        pygame.display.set_mode((600, 400))
        font = pygame_font.Font(None, 24)
        color, text, center, pos = ((160, 200, 250), 'Music', (190, 170), 'midright')
        img1 = font.render(text, True, color)
        img = pygame.Surface(img1.get_size(), depth=32)
        pre_blit_corner_pixel = img.get_at((0, 0))
        img.blit(img1, (0, 0))
        post_blit_corner_pixel = img.get_at((0, 0))
        self.assertEqual(pre_blit_corner_pixel, post_blit_corner_pixel)

    def test_segfault_after_reinit(self):
        """Reinitialization of font module should not cause
        segmentation fault"""
        import gc
        font = pygame_font.Font(None, 20)
        pygame_font.quit()
        pygame_font.init()
        del font
        gc.collect()

    def test_quit(self):
        pygame_font.quit()