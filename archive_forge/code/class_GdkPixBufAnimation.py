from ctypes import *
from pyglet.gl import *
from pyglet.image import *
from pyglet.image.codecs import *
from pyglet.image.codecs import gif
import pyglet.lib
import pyglet.window
class GdkPixBufAnimation:
    """
    Wrapper for a GdkPixBufIter for an animation.
    """

    def __init__(self, loader, anim, gif_delays):
        self._loader = loader
        self._anim = anim
        self._gif_delays = gif_delays
        gdk.g_object_ref(anim)

    def __del__(self):
        if self._anim is not None:
            gdk.g_object_unref(self._anim)

    def __iter__(self):
        time = GTimeVal(0, 0)
        anim_iter = gdkpixbuf.gdk_pixbuf_animation_get_iter(self._anim, byref(time))
        return GdkPixBufAnimationIterator(self._loader, anim_iter, time, self._gif_delays)

    def to_animation(self):
        return Animation(list(self))