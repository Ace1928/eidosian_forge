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
class TextureArray(Texture, UniformTextureSequence):

    def __init__(self, width, height, target, tex_id, max_depth):
        super().__init__(width, height, target, tex_id)
        self.max_depth = max_depth
        self.items = []

    @classmethod
    def create(cls, width, height, internalformat=GL_RGBA, min_filter=None, mag_filter=None, max_depth=256):
        """Create an empty TextureArray.

        You may specify the maximum depth, or layers, the Texture Array should have. This defaults
        to 256, but will be hardware and driver dependent.

        :Parameters:
            `width` : int
                Width of the texture.
            `height` : int
                Height of the texture.
            `internalformat` : int
                GL constant giving the internal format of the texture array; for example, ``GL_RGBA``.
            `min_filter` : int
                The minifaction filter used for this texture array, commonly ``GL_LINEAR`` or ``GL_NEAREST``
            `mag_filter` : int
                The magnification filter used for this texture array, commonly ``GL_LINEAR`` or ``GL_NEAREST``
            `max_depth` : int
                The number of layers in the texture array.

        :rtype: :py:class:`~pyglet.image.TextureArray`

        .. versionadded:: 2.0
        """
        min_filter = min_filter or cls.default_min_filter
        mag_filter = mag_filter or cls.default_mag_filter
        max_depth_limit = get_max_array_texture_layers()
        assert max_depth <= max_depth_limit, 'TextureArray max_depth supported is {}.'.format(max_depth_limit)
        tex_id = GLuint()
        glGenTextures(1, byref(tex_id))
        glBindTexture(GL_TEXTURE_2D_ARRAY, tex_id.value)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, min_filter)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, mag_filter)
        glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, internalformat, width, height, max_depth, 0, internalformat, GL_UNSIGNED_BYTE, 0)
        glFlush()
        texture = cls(width, height, GL_TEXTURE_2D_ARRAY, tex_id.value, max_depth)
        texture.min_filter = min_filter
        texture.mag_filter = mag_filter
        return texture

    def _verify_size(self, image):
        if image.width > self.width or image.height > self.height:
            raise TextureArraySizeExceeded(f'Image ({image.width}x{image.height}) exceeds the size of the TextureArray ({self.width}x{self.height})')

    def add(self, image: pyglet.image.ImageData):
        if len(self.items) >= self.max_depth:
            raise TextureArrayDepthExceeded(f'TextureArray is full.')
        self._verify_size(image)
        start_length = len(self.items)
        item = self.region_class(0, 0, start_length, image.width, image.height, self)
        self.blit_into(image, image.anchor_x, image.anchor_y, start_length)
        self.items.append(item)
        return item

    def allocate(self, *images):
        """Allocates multiple images at once."""
        if len(self.items) + len(images) > self.max_depth:
            raise TextureArrayDepthExceeded('The amount of images being added exceeds the depth of this TextureArray.')
        glBindTexture(self.target, self.id)
        start_length = len(self.items)
        for i, image in enumerate(images):
            self._verify_size(image)
            item = self.region_class(0, 0, start_length + i, image.width, image.height, self)
            self.items.append(item)
            image.blit_to_texture(self.target, self.level, image.anchor_x, image.anchor_y, start_length + i)
        return self.items[start_length:]

    @classmethod
    def create_for_image_grid(cls, grid, internalformat=GL_RGBA):
        texture_array = cls.create(grid[0].width, grid[0].height, internalformat, max_depth=len(grid))
        texture_array.allocate(*grid[:])
        return texture_array

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]

    def __setitem__(self, index, value):
        if type(index) is slice:
            glBindTexture(self.target, self.id)
            for old_item, image in zip(self[index], value):
                self._verify_size(image)
                item = self.region_class(0, 0, old_item.z, image.width, image.height, self)
                image.blit_to_texture(self.target, self.level, image.anchor_x, image.anchor_y, old_item.z)
                self.items[old_item.z] = item
        else:
            self._verify_size(value)
            item = self.region_class(0, 0, index, value.width, value.height, self)
            self.blit_into(value, value.anchor_x, value.anchor_y, index)
            self.items[index] = item

    def __iter__(self):
        return iter(self.items)