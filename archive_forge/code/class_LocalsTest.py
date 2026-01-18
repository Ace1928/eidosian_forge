import unittest
import pygame.constants
import pygame.locals
class LocalsTest(unittest.TestCase):

    def test_locals_has_all_constants(self):
        constants_set = set(pygame.constants.__all__)
        locals_set = set(pygame.locals.__all__)
        self.assertEqual(constants_set - locals_set, set())