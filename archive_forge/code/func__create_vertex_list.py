import sys
import pyglet
from pyglet.gl import *
from pyglet import clock
from pyglet import event
from pyglet import graphics
from pyglet import image
def _create_vertex_list(self):
    texture = self._texture
    self._vertex_list = self.program.vertex_list(1, GL_POINTS, self._batch, self._group, position=('f', (self._x, self._y, self._z)), size=('f', (texture.width, texture.height, texture.anchor_x, texture.anchor_y)), scale=('f', (self._scale_x, self._scale_y)), color=('Bn', self._rgba), texture_uv=('f', texture.uv), rotation=('f', (self._rotation,)))