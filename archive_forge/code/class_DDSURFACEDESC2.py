import struct
import itertools
from pyglet.gl import *
from pyglet.image import CompressedImageData
from pyglet.image import codecs
from pyglet.image.codecs import s3tc, ImageDecodeException
class DDSURFACEDESC2(_FileStruct):
    _fields = [('dwMagic', '4s'), ('dwSize', 'I'), ('dwFlags', 'I'), ('dwHeight', 'I'), ('dwWidth', 'I'), ('dwPitchOrLinearSize', 'I'), ('dwDepth', 'I'), ('dwMipMapCount', 'I'), ('dwReserved1', '44s'), ('ddpfPixelFormat', '32s'), ('dwCaps1', 'I'), ('dwCaps2', 'I'), ('dwCapsReserved', '8s'), ('dwReserved2', 'I')]

    def __init__(self, data):
        super(DDSURFACEDESC2, self).__init__(data)
        self.ddpfPixelFormat = DDPIXELFORMAT(self.ddpfPixelFormat)