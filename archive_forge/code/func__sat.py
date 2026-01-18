import math
from abc import ABC, abstractmethod
import pyglet
from pyglet.gl import GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_BLEND, GL_TRIANGLES
from pyglet.gl import glBlendFunc, glEnable, glDisable
from pyglet.graphics import Batch, Group
from pyglet.math import Vec2
def _sat(vertices, point):
    poly = vertices + [vertices[0]]
    for i in range(len(poly) - 1):
        a, b = (poly[i], poly[i + 1])
        base = Vec2(a[1] - b[1], b[0] - a[0])
        projections = []
        for x, y in poly:
            vec = Vec2(x, y)
            projections.append(base.dot(vec) / abs(base))
        point_proj = base.dot(Vec2(*point)) / abs(base)
        if point_proj < min(projections) or point_proj > max(projections):
            return False
    return True