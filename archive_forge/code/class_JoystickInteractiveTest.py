import unittest
from pygame.tests.test_utils import question, prompt
import pygame
import pygame._sdl2.controller
class JoystickInteractiveTest(unittest.TestCase):
    __tags__ = ['interactive']

    def test_get_count_interactive(self):
        prompt('Please connect any joysticks/controllers now before starting the joystick.get_count() test.')
        pygame.joystick.init()
        count = pygame.joystick.get_count()
        response = question(f'NOTE: Having Steam open may add an extra virtual controller for each joystick/controller physically plugged in.\njoystick.get_count() thinks there is [{count}] joystick(s)/controller(s)connected to this system. Is this correct?')
        self.assertTrue(response)
        if count != 0:
            for x in range(count):
                pygame.joystick.Joystick(x)
            with self.assertRaises(pygame.error):
                pygame.joystick.Joystick(count)
        pygame.joystick.quit()