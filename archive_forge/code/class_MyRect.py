import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
class MyRect(Rect):

    def __init__(self, *args, **kwds):
        super(SubclassTest.MyRect, self).__init__(*args, **kwds)
        self.an_attribute = True