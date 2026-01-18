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
class AbstractImage:
    """Abstract class representing an image.

    :Parameters:
        `width` : int
            Width of image
        `height` : int
            Height of image
        `anchor_x` : int
            X coordinate of anchor, relative to left edge of image data
        `anchor_y` : int
            Y coordinate of anchor, relative to bottom edge of image data
    """
    anchor_x = 0
    anchor_y = 0

    def __init__(self, width, height):
        self.width = width
        self.height = height

    def __repr__(self):
        return '{}(size={}x{})'.format(self.__class__.__name__, self.width, self.height)

    def get_image_data(self):
        """Get an ImageData view of this image.

        Changes to the returned instance may or may not be reflected in this
        image.

        :rtype: :py:class:`~pyglet.image.ImageData`

        .. versionadded:: 1.1
        """
        raise ImageException('Cannot retrieve image data for %r' % self)

    def get_texture(self, rectangle=False):
        """A :py:class:`~pyglet.image.Texture` view of this image.

        :Parameters:
            `rectangle` : bool
                Unused. Kept for compatibility.

                .. versionadded:: 1.1.4.
        :rtype: :py:class:`~pyglet.image.Texture`

        .. versionadded:: 1.1
        """
        raise ImageException('Cannot retrieve texture for %r' % self)

    def get_mipmapped_texture(self):
        """Retrieve a :py:class:`~pyglet.image.Texture` instance with all mipmap levels filled in.

        :rtype: :py:class:`~pyglet.image.Texture`

        .. versionadded:: 1.1
        """
        raise ImageException('Cannot retrieve mipmapped texture for %r' % self)

    def get_region(self, x, y, width, height):
        """Retrieve a rectangular region of this image.

        :Parameters:
            `x` : int
                Left edge of region.
            `y` : int
                Bottom edge of region.
            `width` : int
                Width of region.
            `height` : int
                Height of region.

        :rtype: AbstractImage
        """
        raise ImageException('Cannot get region for %r' % self)

    def save(self, filename=None, file=None, encoder=None):
        """Save this image to a file.

        :Parameters:
            `filename` : str
                Used to set the image file format, and to open the output file
                if `file` is unspecified.
            `file` : file-like object or None
                File to write image data to.
            `encoder` : ImageEncoder or None
                If unspecified, all encoders matching the filename extension
                are tried.  If all fail, the exception from the first one
                attempted is raised.

        """
        if not file:
            file = open(filename, 'wb')
        if encoder:
            encoder.encode(self, filename, file)
        else:
            first_exception = None
            for encoder in _codec_registry.get_encoders(filename):
                try:
                    return encoder.encode(self, filename, file)
                except ImageEncodeException as e:
                    first_exception = first_exception or e
                    file.seek(0)
            if not first_exception:
                raise ImageEncodeException('No image encoders are available')
            raise first_exception

    def blit(self, x, y, z=0):
        """Draw this image to the active framebuffers.

        The image will be drawn with the lower-left corner at
        (``x -`` `anchor_x`, ``y -`` `anchor_y`, ``z``).
        """
        raise ImageException('Cannot blit %r.' % self)

    def blit_into(self, source, x, y, z):
        """Draw `source` on this image.

        `source` will be copied into this image such that its anchor point
        is aligned with the `x` and `y` parameters.  If this image is a 3D
        texture, the `z` coordinate gives the image slice to copy into.

        Note that if `source` is larger than this image (or the positioning
        would cause the copy to go out of bounds) then you must pass a
        region of `source` to this method, typically using get_region().
        """
        raise ImageException('Cannot blit images onto %r.' % self)

    def blit_to_texture(self, target, level, x, y, z=0):
        """Draw this image on the currently bound texture at `target`.

        This image is copied into the texture such that this image's anchor
        point is aligned with the given `x` and `y` coordinates of the
        destination texture.  If the currently bound texture is a 3D texture,
        the `z` coordinate gives the image slice to blit into.
        """
        raise ImageException('Cannot blit %r to a texture.' % self)