import unittest
import pygame
import pygame._sdl2.controller as controller
from pygame.tests.test_utils import prompt, question
class ControllerModuleTest(unittest.TestCase):

    def setUp(self):
        controller.init()

    def tearDown(self):
        controller.quit()

    def test_init(self):
        controller.quit()
        controller.init()
        self.assertTrue(controller.get_init())

    def test_init__multiple(self):
        controller.init()
        controller.init()
        self.assertTrue(controller.get_init())

    def test_quit(self):
        controller.quit()
        self.assertFalse(controller.get_init())

    def test_quit__multiple(self):
        controller.quit()
        controller.quit()
        self.assertFalse(controller.get_init())

    def test_get_init(self):
        self.assertTrue(controller.get_init())

    def test_get_eventstate(self):
        controller.set_eventstate(True)
        self.assertTrue(controller.get_eventstate())
        controller.set_eventstate(False)
        self.assertFalse(controller.get_eventstate())
        controller.set_eventstate(True)

    def test_get_count(self):
        self.assertGreaterEqual(controller.get_count(), 0)

    def test_is_controller(self):
        for i in range(controller.get_count()):
            if controller.is_controller(i):
                c = controller.Controller(i)
                self.assertIsInstance(c, controller.Controller)
                c.quit()
            else:
                with self.assertRaises(pygame._sdl2.sdl2.error):
                    c = controller.Controller(i)
        with self.assertRaises(TypeError):
            controller.is_controller('Test')

    def test_name_forindex(self):
        self.assertIsNone(controller.name_forindex(-1))