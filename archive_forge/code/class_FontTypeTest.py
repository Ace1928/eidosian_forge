from re import T
import sys
import os
import unittest
import pathlib
import platform
import pygame
from pygame import font as pygame_font  # So font can be replaced with ftfont
@unittest.skipIf(IS_PYPY, 'pypy skip known failure')
class FontTypeTest(unittest.TestCase):

    def setUp(self):
        pygame_font.init()

    def tearDown(self):
        pygame_font.quit()

    def test_default_parameters(self):
        f = pygame_font.Font()

    def test_get_ascent(self):
        f = pygame_font.Font(None, 20)
        ascent = f.get_ascent()
        self.assertTrue(isinstance(ascent, int))
        self.assertTrue(ascent > 0)
        s = f.render('X', False, (255, 255, 255))
        self.assertTrue(s.get_size()[1] > ascent)

    def test_get_descent(self):
        f = pygame_font.Font(None, 20)
        descent = f.get_descent()
        self.assertTrue(isinstance(descent, int))
        self.assertTrue(descent < 0)

    def test_get_height(self):
        f = pygame_font.Font(None, 20)
        height = f.get_height()
        self.assertTrue(isinstance(height, int))
        self.assertTrue(height > 0)
        s = f.render('X', False, (255, 255, 255))
        self.assertTrue(s.get_size()[1] == height)

    def test_get_linesize(self):
        f = pygame_font.Font(None, 20)
        linesize = f.get_linesize()
        self.assertTrue(isinstance(linesize, int))
        self.assertTrue(linesize > 0)

    def test_metrics(self):
        f = pygame_font.Font(None, 20)
        um = f.metrics('.')
        bm = f.metrics(b'.')
        self.assertEqual(len(um), 1)
        self.assertEqual(len(bm), 1)
        self.assertIsNotNone(um[0])
        self.assertEqual(um, bm)
        u = 'K'
        b = u.encode('UTF-16')[2:]
        bm = f.metrics(b)
        self.assertEqual(len(bm), 2)
        try:
            um = f.metrics(u)
        except pygame.error:
            pass
        else:
            self.assertEqual(len(um), 1)
            self.assertNotEqual(bm[0], um[0])
            self.assertNotEqual(bm[1], um[0])
        u = '𓀀'
        bm = f.metrics(u)
        self.assertEqual(len(bm), 1)
        self.assertIsNone(bm[0])
        return
        self.fail()

    def test_render(self):
        f = pygame_font.Font(None, 20)
        s = f.render('foo', True, [0, 0, 0], [255, 255, 255])
        s = f.render('xxx', True, [0, 0, 0], [255, 255, 255])
        s = f.render('', True, [0, 0, 0], [255, 255, 255])
        s = f.render('foo', False, [0, 0, 0], [255, 255, 255])
        s = f.render('xxx', False, [0, 0, 0], [255, 255, 255])
        s = f.render('xxx', False, [0, 0, 0])
        s = f.render('   ', False, [0, 0, 0])
        s = f.render('   ', False, [0, 0, 0], [255, 255, 255])
        s = f.render('', False, [0, 0, 0], [255, 255, 255])
        self.assertEqual(s.get_size()[0], 0)
        s = f.render(None, False, [0, 0, 0], [255, 255, 255])
        self.assertEqual(s.get_size()[0], 0)
        self.assertRaises(TypeError, f.render, [], False, [0, 0, 0], [255, 255, 255])
        self.assertRaises(TypeError, f.render, 1, False, [0, 0, 0], [255, 255, 255])
        s = f.render('.', True, [255, 255, 255])
        self.assertEqual(s.get_at((0, 0))[3], 0)
        su = f.render('.', False, [0, 0, 0], [255, 255, 255])
        sb = f.render(b'.', False, [0, 0, 0], [255, 255, 255])
        self.assertTrue(equal_images(su, sb))
        u = 'K'
        b = u.encode('UTF-16')[2:]
        sb = f.render(b, False, [0, 0, 0], [255, 255, 255])
        try:
            su = f.render(u, False, [0, 0, 0], [255, 255, 255])
        except pygame.error:
            pass
        else:
            self.assertFalse(equal_images(su, sb))
        self.assertRaises(ValueError, f.render, b'ab\x00cd', 0, [0, 0, 0])
        self.assertRaises(ValueError, f.render, 'ab\x00cd', 0, [0, 0, 0])

    def test_render_ucs2_ucs4(self):
        """that it renders without raising if there is a new enough SDL_ttf."""
        f = pygame_font.Font(None, 20)
        if hasattr(pygame_font, 'UCS4'):
            ucs_2 = '￮'
            s = f.render(ucs_2, False, [0, 0, 0], [255, 255, 255])
            ucs_4 = '𐀀'
            s = f.render(ucs_4, False, [0, 0, 0], [255, 255, 255])

    def test_set_bold(self):
        f = pygame_font.Font(None, 20)
        self.assertFalse(f.get_bold())
        f.set_bold(True)
        self.assertTrue(f.get_bold())
        f.set_bold(False)
        self.assertFalse(f.get_bold())

    def test_set_italic(self):
        f = pygame_font.Font(None, 20)
        self.assertFalse(f.get_italic())
        f.set_italic(True)
        self.assertTrue(f.get_italic())
        f.set_italic(False)
        self.assertFalse(f.get_italic())

    def test_set_underline(self):
        f = pygame_font.Font(None, 20)
        self.assertFalse(f.get_underline())
        f.set_underline(True)
        self.assertTrue(f.get_underline())
        f.set_underline(False)
        self.assertFalse(f.get_underline())

    def test_set_strikethrough(self):
        if pygame_font.__name__ != 'pygame.ftfont':
            f = pygame_font.Font(None, 20)
            self.assertFalse(f.get_strikethrough())
            f.set_strikethrough(True)
            self.assertTrue(f.get_strikethrough())
            f.set_strikethrough(False)
            self.assertFalse(f.get_strikethrough())

    def test_bold_attr(self):
        f = pygame_font.Font(None, 20)
        self.assertFalse(f.bold)
        f.bold = True
        self.assertTrue(f.bold)
        f.bold = False
        self.assertFalse(f.bold)

    def test_set_italic_property(self):
        f = pygame_font.Font(None, 20)
        self.assertFalse(f.italic)
        f.italic = True
        self.assertTrue(f.italic)
        f.italic = False
        self.assertFalse(f.italic)

    def test_set_underline_property(self):
        f = pygame_font.Font(None, 20)
        self.assertFalse(f.underline)
        f.underline = True
        self.assertTrue(f.underline)
        f.underline = False
        self.assertFalse(f.underline)

    def test_set_strikethrough_property(self):
        if pygame_font.__name__ != 'pygame.ftfont':
            f = pygame_font.Font(None, 20)
            self.assertFalse(f.strikethrough)
            f.strikethrough = True
            self.assertTrue(f.strikethrough)
            f.strikethrough = False
            self.assertFalse(f.strikethrough)

    def test_size(self):
        f = pygame_font.Font(None, 20)
        text = 'Xg'
        size = f.size(text)
        w, h = size
        s = f.render(text, False, (255, 255, 255))
        btext = text.encode('ascii')
        self.assertIsInstance(w, int)
        self.assertIsInstance(h, int)
        self.assertEqual(s.get_size(), size)
        self.assertEqual(f.size(btext), size)
        text = 'K'
        btext = text.encode('UTF-16')[2:]
        bsize = f.size(btext)
        size = f.size(text)
        self.assertNotEqual(size, bsize)

    def test_font_file_not_found(self):
        pygame_font.init()
        self.assertRaises(FileNotFoundError, pygame_font.Font, 'some-fictional-font.ttf', 20)

    def test_load_from_file(self):
        font_name = pygame_font.get_default_font()
        font_path = os.path.join(os.path.split(pygame.__file__)[0], pygame_font.get_default_font())
        f = pygame_font.Font(font_path, 20)

    def test_load_from_file_default(self):
        font_name = pygame_font.get_default_font()
        font_path = os.path.join(os.path.split(pygame.__file__)[0], pygame_font.get_default_font())
        f = pygame_font.Font(font_path)

    def test_load_from_pathlib(self):
        font_name = pygame_font.get_default_font()
        font_path = os.path.join(os.path.split(pygame.__file__)[0], pygame_font.get_default_font())
        f = pygame_font.Font(pathlib.Path(font_path), 20)
        f = pygame_font.Font(pathlib.Path(font_path))

    def test_load_from_pathlib_default(self):
        font_name = pygame_font.get_default_font()
        font_path = os.path.join(os.path.split(pygame.__file__)[0], pygame_font.get_default_font())
        f = pygame_font.Font(pathlib.Path(font_path))

    def test_load_from_file_obj(self):
        font_name = pygame_font.get_default_font()
        font_path = os.path.join(os.path.split(pygame.__file__)[0], pygame_font.get_default_font())
        with open(font_path, 'rb') as f:
            font = pygame_font.Font(f, 20)

    def test_load_from_file_obj_default(self):
        font_name = pygame_font.get_default_font()
        font_path = os.path.join(os.path.split(pygame.__file__)[0], pygame_font.get_default_font())
        with open(font_path, 'rb') as f:
            font = pygame_font.Font(f)

    def test_load_default_font_filename(self):
        f = pygame_font.Font(pygame_font.get_default_font(), 20)

    def test_load_default_font_filename_default(self):
        f = pygame_font.Font(pygame_font.get_default_font())

    def _load_unicode(self, path):
        import shutil
        fdir = str(FONTDIR)
        temp = os.path.join(fdir, path)
        pgfont = os.path.join(fdir, 'test_sans.ttf')
        shutil.copy(pgfont, temp)
        try:
            with open(temp, 'rb') as f:
                pass
        except FileNotFoundError:
            raise unittest.SkipTest('the path cannot be opened')
        try:
            pygame_font.Font(temp, 20)
        finally:
            os.remove(temp)

    def test_load_from_file_unicode_0(self):
        """ASCII string as a unicode object"""
        self._load_unicode('temp_file.ttf')

    def test_load_from_file_unicode_1(self):
        self._load_unicode('你好.ttf')

    def test_load_from_file_bytes(self):
        font_path = os.path.join(os.path.split(pygame.__file__)[0], pygame_font.get_default_font())
        filesystem_encoding = sys.getfilesystemencoding()
        filesystem_errors = 'replace' if sys.platform == 'win32' else 'surrogateescape'
        try:
            font_path = font_path.decode(filesystem_encoding, filesystem_errors)
        except AttributeError:
            pass
        bfont_path = font_path.encode(filesystem_encoding, filesystem_errors)
        f = pygame_font.Font(bfont_path, 20)

    def test_issue_3144(self):
        fpath = os.path.join(FONTDIR, 'PlayfairDisplaySemibold.ttf')
        for size in (60, 40, 10, 20, 70, 45, 50, 10):
            font = pygame_font.Font(fpath, size)
            font.render('WHERE', True, 'black')

    def test_font_set_script(self):
        if pygame_font.__name__ == 'pygame.ftfont':
            return
        font = pygame_font.Font(None, 16)
        ttf_version = pygame_font.get_sdl_ttf_version()
        if ttf_version >= (2, 20, 0):
            self.assertRaises(TypeError, pygame.font.Font.set_script)
            self.assertRaises(TypeError, pygame.font.Font.set_script, font)
            self.assertRaises(TypeError, pygame.font.Font.set_script, 'hey', 'Deva')
            self.assertRaises(TypeError, font.set_script, 1)
            self.assertRaises(TypeError, font.set_script, ['D', 'e', 'v', 'a'])
            self.assertRaises(ValueError, font.set_script, 'too long by far')
            self.assertRaises(ValueError, font.set_script, '')
            self.assertRaises(ValueError, font.set_script, 'a')
            font.set_script('Deva')
        else:
            self.assertRaises(pygame.error, font.set_script, 'Deva')