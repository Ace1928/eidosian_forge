import os.path
from pyglet.util import CodecRegistry, Decoder, Encoder, DecodeException, EncodeException
from pyglet import compat_platform
class ImageDecoder(Decoder):

    def get_animation_file_extensions(self):
        """Return a list of accepted file extensions, e.g. ['.gif', '.flc']
        Lower-case only.
        """
        return []

    def decode(self, filename, file):
        """Decode the given file object and return an instance of `Image`.
        Throws ImageDecodeException if there is an error.  filename
        can be a file type hint.
        """
        raise NotImplementedError()

    def decode_animation(self, filename, file):
        """Decode the given file object and return an instance of :py:class:`~pyglet.image.Animation`.
        Throws ImageDecodeException if there is an error.  filename
        can be a file type hint.
        """
        raise ImageDecodeException('This decoder cannot decode animations.')

    def __repr__(self):
        return '{0}{1}'.format(self.__class__.__name__, self.get_animation_file_extensions() + self.get_file_extensions())