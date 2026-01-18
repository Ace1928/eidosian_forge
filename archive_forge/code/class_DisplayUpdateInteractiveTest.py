import unittest
import os
import sys
import time
import pygame, pygame.transform
from pygame.tests.test_utils import question
from pygame import display
class DisplayUpdateInteractiveTest(DisplayUpdateTest):
    """Because we want these tests to run as interactive and not interactive."""
    __tags__ = ['interactive']

    def question(self, qstr):
        """since this is the interactive sublcass we ask a question."""
        question(qstr)