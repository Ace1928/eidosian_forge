import unittest
from pygame.tests.test_utils import question, prompt
import pygame
import pygame._sdl2.controller
class JoystickModuleTest(unittest.TestCase):

    def test_get_init(self):

        def error_check_get_init():
            try:
                pygame.joystick.get_count()
            except pygame.error:
                return False
            return True
        self.assertEqual(pygame.joystick.get_init(), False)
        pygame.joystick.init()
        self.assertEqual(pygame.joystick.get_init(), error_check_get_init())
        pygame.joystick.quit()
        self.assertEqual(pygame.joystick.get_init(), error_check_get_init())
        pygame.joystick.init()
        pygame.joystick.init()
        self.assertEqual(pygame.joystick.get_init(), error_check_get_init())
        pygame.joystick.quit()
        self.assertEqual(pygame.joystick.get_init(), error_check_get_init())
        pygame.joystick.quit()
        self.assertEqual(pygame.joystick.get_init(), error_check_get_init())
        for i in range(100):
            pygame.joystick.init()
        self.assertEqual(pygame.joystick.get_init(), error_check_get_init())
        pygame.joystick.quit()
        self.assertEqual(pygame.joystick.get_init(), error_check_get_init())
        for i in range(100):
            pygame.joystick.quit()
        self.assertEqual(pygame.joystick.get_init(), error_check_get_init())

    def test_init(self):
        """
        This unit test is for joystick.init()
        It was written to help reduce maintenance costs
        and to help test against changes to the code or
        different platforms.
        """
        pygame.quit()
        pygame.init()
        self.assertEqual(pygame.joystick.get_init(), True)
        pygame._sdl2.controller.quit()
        pygame.joystick.quit()
        with self.assertRaises(pygame.error):
            pygame.joystick.get_count()
        iterations = 20
        for i in range(iterations):
            pygame.joystick.init()
        self.assertEqual(pygame.joystick.get_init(), True)
        self.assertIsNotNone(pygame.joystick.get_count())

    def test_quit(self):
        """Test if joystick.quit works."""
        pygame.joystick.init()
        self.assertIsNotNone(pygame.joystick.get_count())
        pygame.joystick.quit()
        with self.assertRaises(pygame.error):
            pygame.joystick.get_count()

    def test_get_count(self):
        pygame.joystick.init()
        try:
            count = pygame.joystick.get_count()
            self.assertGreaterEqual(count, 0, 'joystick.get_count() must return a value >= 0')
        finally:
            pygame.joystick.quit()