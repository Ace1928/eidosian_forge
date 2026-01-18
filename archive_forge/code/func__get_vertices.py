import math
from abc import ABC, abstractmethod
import pyglet
from pyglet.gl import GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_BLEND, GL_TRIANGLES
from pyglet.gl import glBlendFunc, glEnable, glDisable
from pyglet.graphics import Batch, Group
from pyglet.math import Vec2
def _get_vertices(self):
    if not self._visible:
        return (0, 0) * self._num_verts
    else:
        trans_x, trans_y = self._coordinates[0]
        trans_x += self._anchor_x
        trans_y += self._anchor_y
        coords = [[x - trans_x, y - trans_y] for x, y in self._coordinates]
        triangles = []
        prev_miter = None
        prev_scale = None
        for i in range(len(coords) - 1):
            prev_point = None
            next_point = None
            if i > 0:
                prev_point = coords[i - 1]
            if i + 2 < len(coords):
                next_point = coords[i + 2]
            prev_miter, prev_scale, *segment = _get_segment(prev_point, coords[i], coords[i + 1], next_point, self._thickness, prev_miter, prev_scale)
            triangles.extend(segment)
        return triangles