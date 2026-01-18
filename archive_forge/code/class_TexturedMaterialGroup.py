import pyglet
from pyglet import gl
from pyglet import graphics
from pyglet.gl import current_context
from pyglet.math import Mat4, Vec3
from pyglet.graphics import shader
from .codecs import registry as _codec_registry
from .codecs import add_default_codecs as _add_default_codecs
class TexturedMaterialGroup(BaseMaterialGroup):
    default_vert_src = '#version 330 core\n    in vec3 position;\n    in vec3 normals;\n    in vec2 tex_coords;\n    in vec4 colors;\n\n    out vec4 vertex_colors;\n    out vec3 vertex_normals;\n    out vec2 texture_coords;\n    out vec3 vertex_position;\n\n    uniform WindowBlock\n    {\n        mat4 projection;\n        mat4 view;\n    } window;\n\n    uniform mat4 model;\n\n    void main()\n    {\n        vec4 pos = window.view * model * vec4(position, 1.0);\n        gl_Position = window.projection * pos;\n        mat3 normal_matrix = transpose(inverse(mat3(model)));\n\n        vertex_position = pos.xyz;\n        vertex_colors = colors;\n        texture_coords = tex_coords;\n        vertex_normals = normal_matrix * normals;\n    }\n    '
    default_frag_src = '#version 330 core\n    in vec4 vertex_colors;\n    in vec3 vertex_normals;\n    in vec2 texture_coords;\n    in vec3 vertex_position;\n    out vec4 final_colors;\n\n    uniform sampler2D our_texture;\n\n    void main()\n    {\n        float l = dot(normalize(-vertex_position), normalize(vertex_normals));\n        final_colors = (texture(our_texture, texture_coords) * vertex_colors) * l * 1.2;\n    }\n    '

    def __init__(self, material, program, texture, order=0, parent=None):
        super().__init__(material, program, order, parent)
        self.texture = texture

    def set_state(self):
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(self.texture.target, self.texture.id)
        self.program.use()
        self.program['model'] = self.matrix

    def __hash__(self):
        return hash((self.texture.target, self.texture.id, self.program, self.order, self.parent))

    def __eq__(self, other):
        return self.__class__ is other.__class__ and self.material == other.material and (self.texture.target == other.texture.target) and (self.texture.id == other.texture.id) and (self.program == other.program) and (self.order == other.order) and (self.parent == other.parent)