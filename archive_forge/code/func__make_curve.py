import math
from abc import ABC, abstractmethod
import pyglet
from pyglet.gl import GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_BLEND, GL_TRIANGLES
from pyglet.gl import glBlendFunc, glEnable, glDisable
from pyglet.graphics import Batch, Group
from pyglet.math import Vec2
def _make_curve(self, t):
    n = len(self._points) - 1
    p = [0, 0]
    for i in range(n + 1):
        m = math.comb(n, i) * (1 - t) ** (n - i) * t ** i
        p[0] += m * self._points[i][0]
        p[1] += m * self._points[i][1]
    return p