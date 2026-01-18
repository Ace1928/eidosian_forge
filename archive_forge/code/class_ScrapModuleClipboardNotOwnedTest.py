import os
import sys
import unittest
from pygame.tests.test_utils import trunk_relative_path
import pygame
from pygame import scrap
class ScrapModuleClipboardNotOwnedTest(unittest.TestCase):
    """Test the scrap module's functionality when the pygame application is
    not the current owner of the clipboard.

    A separate class is used to prevent tests that acquire the clipboard from
    interfering with these tests.
    """

    @classmethod
    def setUpClass(cls):
        pygame.display.init()
        pygame.display.set_mode((1, 1))
        scrap.init()

    @classmethod
    def tearDownClass(cls):
        pygame.quit()
        pygame.display.quit()

    def _skip_if_clipboard_owned(self):
        if not scrap.lost():
            self.skipTest('requires the pygame application to not own the clipboard')

    def test_get__not_owned(self):
        """Ensures get works when there is no data of the requested type
        in the clipboard and the clipboard is not owned by the pygame
        application.
        """
        self._skip_if_clipboard_owned()
        DATA_TYPE = 'test_get__not_owned'
        data = scrap.get(DATA_TYPE)
        self.assertIsNone(data)

    def test_get_types__not_owned(self):
        """Ensures get_types works when the clipboard is not owned
        by the pygame application.
        """
        self._skip_if_clipboard_owned()
        data_types = scrap.get_types()
        self.assertIsInstance(data_types, list)

    def test_contains__not_owned(self):
        """Ensures contains works when the clipboard is not owned
        by the pygame application.
        """
        self._skip_if_clipboard_owned()
        DATA_TYPE = 'test_contains__not_owned'
        contains = scrap.contains(DATA_TYPE)
        self.assertFalse(contains)

    def test_lost__not_owned(self):
        """Ensures lost works when the clipboard is not owned
        by the pygame application.
        """
        self._skip_if_clipboard_owned()
        lost = scrap.lost()
        self.assertTrue(lost)