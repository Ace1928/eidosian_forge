from ctypes import *
from pyglet.gl import *
from pyglet.image import *
from pyglet.image.codecs import *
from pyglet.image.codecs import gif
import pyglet.lib
import pyglet.window
def _cancel_load(self):
    assert not self.closed
    gdkpixbuf.gdk_pixbuf_loader_close(self._loader, None)
    self.closed = True