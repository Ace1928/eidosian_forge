import os.path
from pyglet.util import CodecRegistry, Decoder, Encoder, DecodeException, EncodeException
from pyglet import compat_platform
def add_default_codecs():
    try:
        from pyglet.image.codecs import dds
        registry.add_encoders(dds)
        registry.add_decoders(dds)
    except ImportError:
        pass
    if compat_platform == 'darwin':
        try:
            from pyglet.image.codecs import quartz
            registry.add_encoders(quartz)
            registry.add_decoders(quartz)
        except ImportError:
            pass
    if compat_platform in ('win32', 'cygwin'):
        from pyglet.libs.win32.constants import WINDOWS_7_OR_GREATER
        if WINDOWS_7_OR_GREATER:
            try:
                from pyglet.image.codecs import wic
                registry.add_encoders(wic)
                registry.add_decoders(wic)
            except ImportError:
                pass
    if compat_platform in ('win32', 'cygwin'):
        try:
            from pyglet.image.codecs import gdiplus
            registry.add_encoders(gdiplus)
            registry.add_decoders(gdiplus)
        except ImportError:
            pass
    if compat_platform.startswith('linux'):
        try:
            from pyglet.image.codecs import gdkpixbuf2
            registry.add_encoders(gdkpixbuf2)
            registry.add_decoders(gdkpixbuf2)
        except ImportError:
            pass
    try:
        from pyglet.image.codecs import pil
        registry.add_encoders(pil)
        registry.add_decoders(pil)
    except ImportError:
        pass
    try:
        from pyglet.image.codecs import png
        registry.add_encoders(png)
        registry.add_decoders(png)
    except ImportError:
        pass
    try:
        from pyglet.image.codecs import bmp
        registry.add_encoders(bmp)
        registry.add_decoders(bmp)
    except ImportError:
        pass