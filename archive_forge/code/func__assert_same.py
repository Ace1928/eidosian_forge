import os
import unittest
from pygame.tests import test_utils
from pygame.tests.test_utils import (
import pygame
from pygame.locals import *
from pygame.bufferproxy import BufferProxy
import platform
import gc
import weakref
import ctypes
def _assert_same(self, a, b):
    w, h = a.get_size()
    for x in range(w):
        for y in range(h):
            self.assertEqual(a.get_at((x, y)), b.get_at((x, y)), '%s != %s, bpp: %i' % (a.get_at((x, y)), b.get_at((x, y)), a.get_bitsize()))