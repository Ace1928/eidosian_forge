import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
class _ObjectWithRectProperty:

    def __init__(self, r):
        self._rect = r

    @property
    def rect(self):
        return self._rect