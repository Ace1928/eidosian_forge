import struct
import itertools
from pyglet.gl import *
from pyglet.image import CompressedImageData
from pyglet.image import codecs
from pyglet.image.codecs import s3tc, ImageDecodeException
class _FileStruct:
    _fields = []

    def __init__(self, data):
        if len(data) < self.get_size():
            raise ImageDecodeException('Not a DDS file')
        items = struct.unpack(self.get_format(), data)
        for field, value in itertools.zip_longest(self._fields, items, fillvalue=None):
            setattr(self, field[0], value)

    def __repr__(self):
        name = self.__class__.__name__
        return '%s(%s)' % (name, (', \n%s' % (' ' * (len(name) + 1))).join(['%s = %s' % (field[0], repr(getattr(self, field[0]))) for field in self._fields]))

    @classmethod
    def get_format(cls):
        return '<' + ''.join([f[1] for f in cls._fields])

    @classmethod
    def get_size(cls):
        return struct.calcsize(cls.get_format())