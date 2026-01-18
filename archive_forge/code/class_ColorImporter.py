import math
import operator
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame
from pygame.colordict import THECOLORS
class ColorImporter(buftools.Importer):

    def __init__(self, color, flags):
        super().__init__(color, flags)
        self.items = cast(self.buf, POINTER(c_uint8))

    def __getitem__(self, index):
        if 0 <= index < 4:
            return self.items[index]
        raise IndexError(f'valid index values are between 0 and 3: got {index}')

    def __setitem__(self, index, value):
        if 0 <= index < 4:
            self.items[index] = value
        else:
            raise IndexError(f'valid index values are between 0 and 3: got {index}')