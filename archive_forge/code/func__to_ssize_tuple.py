import a new buffer interface.
import pygame
import pygame.newbuffer
from pygame.newbuffer import (
import unittest
import ctypes
import operator
from functools import reduce
def _to_ssize_tuple(self, addr):
    from ctypes import cast, POINTER, c_ssize_t
    if addr is None:
        return None
    return tuple(cast(addr, POINTER(c_ssize_t))[0:self._view.ndim])