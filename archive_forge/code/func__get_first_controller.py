import unittest
import pygame
import pygame._sdl2.controller as controller
from pygame.tests.test_utils import prompt, question
def _get_first_controller(self):
    for i in range(controller.get_count()):
        if controller.is_controller(i):
            return controller.Controller(i)