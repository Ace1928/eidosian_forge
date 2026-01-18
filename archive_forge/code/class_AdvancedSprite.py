import sys
import pyglet
from pyglet.gl import *
from pyglet import clock
from pyglet import event
from pyglet import graphics
from pyglet import image
class AdvancedSprite(pyglet.sprite.Sprite):
    """Is a sprite that lets you change the shader program during initialization and after
    For advanced users who understand shaders."""

    def __init__(self, img, x=0, y=0, z=0, blend_src=GL_SRC_ALPHA, blend_dest=GL_ONE_MINUS_SRC_ALPHA, batch=None, group=None, subpixel=False, program=None):
        self._program = program
        if not program:
            if isinstance(img, image.TextureArrayRegion):
                self._program = get_default_array_shader()
            else:
                self._program = get_default_shader()
        super().__init__(img, x, y, z, blend_src, blend_dest, batch, group, subpixel)

    @property
    def program(self):
        return self._program

    @program.setter
    def program(self, program):
        if self._program == program:
            return
        self._group = self.group_class(self._texture, self._group.blend_src, self._group.blend_dest, program, self._group)
        self._batch.migrate(self._vertex_list, GL_TRIANGLES, self._group, self._batch)
        self._program = program