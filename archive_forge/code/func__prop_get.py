import a new buffer interface.
import pygame
import pygame.newbuffer
from pygame.newbuffer import (
import unittest
import ctypes
import operator
from functools import reduce
def _prop_get(fn):
    return property(fn)