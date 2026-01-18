import sys
import pytest
import itertools
from threading import Thread
from kivy.tests.common import GraphicUnitTest, requires_graphics
def _filtered_points(self, points):
    index = 0
    p = self._convert_points(points)
    if len(p) < 6:
        return []
    while index < len(p) - 2:
        x1, y1 = (p[index], p[index + 1])
        x2, y2 = (p[index + 2], p[index + 3])
        if abs(x2 - x1) < 1.0 and abs(y2 - y1) < 1.0:
            del p[index + 2:index + 4]
        else:
            index += 2
    if abs(p[0] - p[-2]) < 1.0 and abs(p[1] - p[-1]) < 1.0:
        del p[:2]
    return p