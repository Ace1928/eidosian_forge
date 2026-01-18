import sys
import pyglet
from pyglet.gl import *
from pyglet import clock
from pyglet import event
from pyglet import graphics
from pyglet import image
def _set_texture(self, texture):
    if texture.id is not self._texture.id:
        self._group = self._group.__class__(texture, self._group.blend_src, self._group.blend_dest, self._group.program, self._group.parent)
        self._vertex_list.delete()
        self._texture = texture
        self._create_vertex_list()
    else:
        self._vertex_list.texture_uv[:] = texture.uv
    self._texture = texture