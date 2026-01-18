import warnings
from ctypes import *
from weakref import proxy
import pyglet
from pyglet.gl import *
from pyglet.graphics.vertexbuffer import BufferObject
class ShaderSource:
    """GLSL source container for making source parsing simpler.

    We support locating out attributes and applying #defines values.

    NOTE: We do assume the source is neat enough to be parsed
    this way and don't contain several statements in one line.
    """

    def __init__(self, source: str, source_type: GLenum):
        """Create a shader source wrapper."""
        self._lines = source.strip().splitlines()
        self._type = source_type
        if not self._lines:
            raise ShaderException('Shader source is empty')
        self._version = self._find_glsl_version()
        if pyglet.gl.current_context.get_info().get_opengl_api() == 'gles':
            self._lines[0] = '#version 310 es'
            self._lines.insert(1, 'precision mediump float;')
            if self._type == GL_GEOMETRY_SHADER:
                self._lines.insert(1, '#extension GL_EXT_geometry_shader : require')
            if self._type == GL_COMPUTE_SHADER:
                self._lines.insert(1, 'precision mediump image2D;')
            self._version = self._find_glsl_version()

    def validate(self) -> str:
        """Return the validated shader source."""
        return '\n'.join(self._lines)

    def _find_glsl_version(self) -> int:
        if self._lines[0].strip().startswith('#version'):
            try:
                return int(self._lines[0].split()[1])
            except (ValueError, IndexError):
                pass
        source = '\n'.join((f'{str(i + 1).zfill(3)}: {line} ' for i, line in enumerate(self._lines)))
        raise ShaderException(f'Cannot find #version flag in shader source. A #version statement is required on the first line.\n------------------------------------\n{source}')