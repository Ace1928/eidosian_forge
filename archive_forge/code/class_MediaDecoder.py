import warnings
from pyglet.util import CodecRegistry, Decoder, Encoder
from .base import *
import pyglet
class MediaDecoder(Decoder):

    def decode(self, filename, file, streaming):
        """Read the given file object and return an instance of `Source`
        or `StreamingSource`. 
        Throws DecodeException if there is an error.  `filename`
        can be a file type hint.
        """
        raise NotImplementedError()