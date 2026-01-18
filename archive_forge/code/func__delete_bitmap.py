from pyglet.libs.win32.com import pIUnknown
from pyglet.image import *
from pyglet.image.codecs import *
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
from pyglet.libs.win32 import _kernel32 as kernel32
from pyglet.libs.win32 import _ole32 as ole32
def _delete_bitmap(self, bitmap):
    gdiplus.GdipDisposeImage(bitmap)
    self.stream.Release()