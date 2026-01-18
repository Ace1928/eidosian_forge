import ctypes
import weakref
import pyglet
from pyglet.gl import *
from pyglet.graphics import shader, vertexdomain
from pyglet.graphics.vertexarray import VertexArray
from pyglet.graphics.vertexbuffer import BufferObject
class ShaderGroup(Group):
    """A group that enables and binds a ShaderProgram.
    """

    def __init__(self, program, order=0, parent=None):
        super().__init__(order, parent)
        self.program = program

    def set_state(self):
        self.program.use()

    def unset_state(self):
        self.program.stop()

    def __eq__(self, other):
        return self.__class__ is other.__class__ and self._order == other.order and (self.program == other.program) and (self.parent == other.parent)

    def __hash__(self):
        return hash((self._order, self.parent, self.program))