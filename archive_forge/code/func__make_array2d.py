import unittest
import platform
from numpy import (
import pygame
from pygame.locals import *
import pygame.surfarray
def _make_array2d(self, dtype):
    return zeros(self.surf_size, dtype)