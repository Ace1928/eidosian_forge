import warnings
from ctypes import *
from weakref import proxy
import pyglet
from pyglet.gl import *
from pyglet.graphics.vertexbuffer import BufferObject
def _find_glsl_version(self) -> int:
    if self._lines[0].strip().startswith('#version'):
        try:
            return int(self._lines[0].split()[1])
        except (ValueError, IndexError):
            pass
    source = '\n'.join((f'{str(i + 1).zfill(3)}: {line} ' for i, line in enumerate(self._lines)))
    raise ShaderException(f'Cannot find #version flag in shader source. A #version statement is required on the first line.\n------------------------------------\n{source}')