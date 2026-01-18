from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
class SubMaskCopy(SubMask):
    """Subclass of the Mask class to help test copying subclasses."""

    def copy(self):
        mask_copy = super().copy()
        mask_copy.test_attribute = self.test_attribute
        return mask_copy