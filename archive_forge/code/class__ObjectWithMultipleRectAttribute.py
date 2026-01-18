import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
class _ObjectWithMultipleRectAttribute:

    def __init__(self, r1, r2, r3):
        self.rect1 = r1
        self.rect2 = r2
        self.rect3 = r3