import re
from base64 import b64decode
import imghdr
from kivy.event import EventDispatcher
from kivy.core import core_register_libs
from kivy.logger import Logger
from kivy.cache import Cache
from kivy.clock import Clock
from kivy.atlas import Atlas
from kivy.resources import resource_find
from kivy.utils import platform
from kivy.compat import string_types
from kivy.setupconfig import USE_SDL2
import zipfile
from io import BytesIO
from os import environ
from kivy.graphics.texture import Texture, TextureRegion
class ImageData(object):
    """Container for images and mipmap images.
    The container will always have at least the mipmap level 0.
    """
    __slots__ = ('fmt', 'mipmaps', 'source', 'flip_vertical', 'source_image')
    _supported_fmts = ('rgb', 'bgr', 'rgba', 'bgra', 'argb', 'abgr', 's3tc_dxt1', 's3tc_dxt3', 's3tc_dxt5', 'pvrtc_rgb2', 'pvrtc_rgb4', 'pvrtc_rgba2', 'pvrtc_rgba4', 'etc1_rgb8')

    def __init__(self, width, height, fmt, data, source=None, flip_vertical=True, source_image=None, rowlength=0):
        assert fmt in ImageData._supported_fmts
        self.fmt = fmt
        self.mipmaps = {}
        self.add_mipmap(0, width, height, data, rowlength)
        self.source = source
        self.flip_vertical = flip_vertical
        self.source_image = source_image

    def release_data(self):
        mm = self.mipmaps
        for item in mm.values():
            item[2] = None
            self.source_image = None

    @property
    def width(self):
        """Image width in pixels.
        (If the image is mipmapped, it will use the level 0)
        """
        return self.mipmaps[0][0]

    @property
    def height(self):
        """Image height in pixels.
        (If the image is mipmapped, it will use the level 0)
        """
        return self.mipmaps[0][1]

    @property
    def data(self):
        """Image data.
        (If the image is mipmapped, it will use the level 0)
        """
        return self.mipmaps[0][2]

    @property
    def rowlength(self):
        """Image rowlength.
        (If the image is mipmapped, it will use the level 0)

        .. versionadded:: 1.9.0
        """
        return self.mipmaps[0][3]

    @property
    def size(self):
        """Image (width, height) in pixels.
        (If the image is mipmapped, it will use the level 0)
        """
        mm = self.mipmaps[0]
        return (mm[0], mm[1])

    @property
    def have_mipmap(self):
        return len(self.mipmaps) > 1

    def __repr__(self):
        return '<ImageData width=%d height=%d fmt=%s source=%r with %d images>' % (self.width, self.height, self.fmt, self.source, len(self.mipmaps))

    def add_mipmap(self, level, width, height, data, rowlength):
        """Add a image for a specific mipmap level.

        .. versionadded:: 1.0.7
        """
        self.mipmaps[level] = [int(width), int(height), data, rowlength]

    def get_mipmap(self, level):
        """Get the mipmap image at a specific level if it exists

        .. versionadded:: 1.0.7
        """
        if level == 0:
            return (self.width, self.height, self.data, self.rowlength)
        assert level < len(self.mipmaps)
        return self.mipmaps[level]

    def iterate_mipmaps(self):
        """Iterate over all mipmap images available.

        .. versionadded:: 1.0.7
        """
        mm = self.mipmaps
        for x in range(len(mm)):
            item = mm.get(x, None)
            if item is None:
                raise Exception('Invalid mipmap level, found empty one')
            yield (x, item[0], item[1], item[2], item[3])