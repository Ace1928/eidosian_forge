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
def _assert_surface(self, surf, palette=None, msg=''):
    if palette is None:
        palette = self._test_palette
    if surf.get_bitsize() == 16:
        palette = [surf.unmap_rgb(surf.map_rgb(c)) for c in palette]
    for posn, i in self._test_points:
        self.assertEqual(surf.get_at(posn), palette[i], '%s != %s: flags: %i, bpp: %i, posn: %s%s' % (surf.get_at(posn), palette[i], surf.get_flags(), surf.get_bitsize(), posn, msg))