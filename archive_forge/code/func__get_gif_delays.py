from ctypes import *
from pyglet.gl import *
from pyglet.image import *
from pyglet.image.codecs import *
from pyglet.image.codecs import gif
import pyglet.lib
import pyglet.window
def _get_gif_delays(self):
    assert self._file is not None
    self._file.seek(0)
    gif_stream = gif.read(self._file)
    return [image.delay for image in gif_stream.images]