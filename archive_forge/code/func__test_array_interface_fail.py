import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
def _test_array_interface_fail(self, a):
    self.assertRaises(ValueError, mixer.Sound, array=a)