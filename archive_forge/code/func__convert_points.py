import sys
import pytest
import itertools
from threading import Thread
from kivy.tests.common import GraphicUnitTest, requires_graphics
def _convert_points(self, points):
    if points and isinstance(points[0], (list, tuple)):
        return list(itertools.chain(*points))
    else:
        return list(points)