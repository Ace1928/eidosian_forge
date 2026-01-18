from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
class SubMaskDunderCopy(SubMask):
    """Subclass of the Mask class to help test copying subclasses."""

    def __copy__(self):
        mask_copy = super().__copy__()
        mask_copy.test_attribute = self.test_attribute
        return mask_copy