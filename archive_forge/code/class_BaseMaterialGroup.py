import pyglet
from pyglet import gl
from pyglet import graphics
from pyglet.gl import current_context
from pyglet.math import Mat4, Vec3
from pyglet.graphics import shader
from .codecs import registry as _codec_registry
from .codecs import add_default_codecs as _add_default_codecs
class BaseMaterialGroup(graphics.Group):
    default_vert_src = None
    default_frag_src = None
    matrix = Mat4()

    def __init__(self, material, program, order=0, parent=None):
        super().__init__(order, parent)
        self.material = material
        self.program = program