import re
import weakref
from ctypes import *
from io import open, BytesIO
import pyglet
from pyglet.gl import *
from pyglet.gl import gl_info
from pyglet.util import asbytes
from .codecs import ImageEncodeException, ImageDecodeException
from .codecs import registry as _codec_registry
from .codecs import add_default_codecs as _add_default_codecs
from .animation import Animation, AnimationFrame
from .buffer import *
from . import atlas
class CompressedImageData(AbstractImage):
    """Image representing some compressed data suitable for direct uploading
    to driver.
    """
    _current_texture = None
    _current_mipmap_texture = None

    def __init__(self, width, height, gl_format, data, extension=None, decoder=None):
        """Construct a CompressedImageData with the given compressed data.

        :Parameters:
            `width` : int
                Width of image
            `height` : int
                Height of image
            `gl_format` : int
                GL constant giving format of compressed data; for example,
                ``GL_COMPRESSED_RGBA_S3TC_DXT5_EXT``.
            `data` : sequence
                String or array/list of bytes giving compressed image data.
            `extension` : str or None
                If specified, gives the name of a GL extension to check for
                before creating a texture.
            `decoder` : function(data, width, height) -> AbstractImage
                A function to decode the compressed data, to be used if the
                required extension is not present.

        """
        super().__init__(width, height)
        self.data = data
        self.gl_format = gl_format
        self.extension = extension
        self.decoder = decoder
        self.mipmap_data = []

    def set_mipmap_data(self, level, data):
        """Set data for a mipmap level.

        Supplied data gives a compressed image for the given mipmap level.
        The image must be of the correct dimensions for the level
        (i.e., width >> level, height >> level); but this is not checked.  If
        any mipmap levels are specified, they are used; otherwise, mipmaps for
        `mipmapped_texture` are generated automatically.

        :Parameters:
            `level` : int
                Level of mipmap image to set.
            `data` : sequence
                String or array/list of bytes giving compressed image data.
                Data must be in same format as specified in constructor.

        """
        self.mipmap_data += [None] * (level - len(self.mipmap_data))
        self.mipmap_data[level - 1] = data

    def _have_extension(self):
        return self.extension is None or gl_info.have_extension(self.extension)

    def _verify_driver_supported(self):
        """Assert that the extension required for this image data is
        supported.

        Raises `ImageException` if not.
        """
        if not self._have_extension():
            raise ImageException('%s is required to decode %r' % (self.extension, self))

    def get_texture(self, rectangle=False):
        if rectangle:
            raise ImageException('Compressed texture rectangles not supported')
        if self._current_texture:
            return self._current_texture
        texture = Texture.create(self.width, self.height, GL_TEXTURE_2D, None)
        if self.anchor_x or self.anchor_y:
            texture.anchor_x = self.anchor_x
            texture.anchor_y = self.anchor_y
        glBindTexture(texture.target, texture.id)
        glTexParameteri(texture.target, GL_TEXTURE_MIN_FILTER, texture.min_filter)
        glTexParameteri(texture.target, GL_TEXTURE_MAG_FILTER, texture.mag_filter)
        if self._have_extension():
            glCompressedTexImage2D(texture.target, texture.level, self.gl_format, self.width, self.height, 0, len(self.data), self.data)
        else:
            image = self.decoder(self.data, self.width, self.height)
            texture = image.get_texture()
            assert texture.width == self.width
            assert texture.height == self.height
        glFlush()
        self._current_texture = texture
        return texture

    def get_mipmapped_texture(self):
        if self._current_mipmap_texture:
            return self._current_mipmap_texture
        if not self._have_extension():
            return self.get_texture()
        texture = Texture.create(self.width, self.height, GL_TEXTURE_2D, None)
        if self.anchor_x or self.anchor_y:
            texture.anchor_x = self.anchor_x
            texture.anchor_y = self.anchor_y
        glBindTexture(texture.target, texture.id)
        glTexParameteri(texture.target, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        if not self.mipmap_data:
            glGenerateMipmap(texture.target)
        glCompressedTexImage2D(texture.target, texture.level, self.gl_format, self.width, self.height, 0, len(self.data), self.data)
        width, height = (self.width, self.height)
        level = 0
        for data in self.mipmap_data:
            width >>= 1
            height >>= 1
            level += 1
            glCompressedTexImage2D(texture.target, level, self.gl_format, width, height, 0, len(data), data)
        glFlush()
        self._current_mipmap_texture = texture
        return texture

    def blit_to_texture(self, target, level, x, y, z):
        self._verify_driver_supported()
        if target == GL_TEXTURE_3D:
            glCompressedTexSubImage3D(target, level, x - self.anchor_x, y - self.anchor_y, z, self.width, self.height, 1, self.gl_format, len(self.data), self.data)
        else:
            glCompressedTexSubImage2D(target, level, x - self.anchor_x, y - self.anchor_y, self.width, self.height, self.gl_format, len(self.data), self.data)