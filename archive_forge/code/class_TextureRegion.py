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
class TextureRegion(Texture):
    """A rectangular region of a texture, presented as if it were a separate texture.
    """

    def __init__(self, x, y, z, width, height, owner):
        super().__init__(width, height, owner.target, owner.id)
        self.x = x
        self.y = y
        self.z = z
        self.owner = owner
        owner_u1 = owner.tex_coords[0]
        owner_v1 = owner.tex_coords[1]
        owner_u2 = owner.tex_coords[3]
        owner_v2 = owner.tex_coords[7]
        scale_u = owner_u2 - owner_u1
        scale_v = owner_v2 - owner_v1
        u1 = x / owner.width * scale_u + owner_u1
        v1 = y / owner.height * scale_v + owner_v1
        u2 = (x + width) / owner.width * scale_u + owner_u1
        v2 = (y + height) / owner.height * scale_v + owner_v1
        r = z / owner.images + owner.tex_coords[2]
        self.tex_coords = (u1, v1, r, u2, v1, r, u2, v2, r, u1, v2, r)

    def get_image_data(self):
        image_data = self.owner.get_image_data(self.z)
        return image_data.get_region(self.x, self.y, self.width, self.height)

    def get_region(self, x, y, width, height):
        x += self.x
        y += self.y
        region = self.region_class(x, y, self.z, width, height, self.owner)
        region._set_tex_coords_order(*self.tex_coords_order)
        return region

    def blit_into(self, source, x, y, z):
        self.owner.blit_into(source, x + self.x, y + self.y, z + self.z)

    def __repr__(self):
        return '{}(id={}, size={}x{}, owner={}x{})'.format(self.__class__.__name__, self.id, self.width, self.height, self.owner.width, self.owner.height)

    def delete(self):
        """Deleting a TextureRegion has no effect. Operate on the owning
        texture instead.
        """
        pass

    def __del__(self):
        pass