import math
import operator
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame
from pygame.colordict import THECOLORS
class MyColor(pygame.Color):

    def __init__(self, *args, **kwds):
        super(SubclassTest.MyColor, self).__init__(*args, **kwds)
        self.an_attribute = True