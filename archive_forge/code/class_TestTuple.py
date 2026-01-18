import math
import operator
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame
from pygame.colordict import THECOLORS
class TestTuple(tuple):

    def __eq__(self, other):
        return -1

    def __ne__(self, other):
        return -2