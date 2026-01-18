import os
import sys
import unittest
from pygame.tests.test_utils import trunk_relative_path
import pygame
from pygame import scrap
def _skip_if_clipboard_owned(self):
    if not scrap.lost():
        self.skipTest('requires the pygame application to not own the clipboard')