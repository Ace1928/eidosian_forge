import os.path
from pyglet.util import CodecRegistry, Decoder, Encoder, DecodeException, EncodeException
from pyglet import compat_platform
class ImageDecodeException(DecodeException):
    pass