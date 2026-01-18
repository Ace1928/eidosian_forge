import struct
import itertools
from pyglet.gl import *
from pyglet.image import CompressedImageData
from pyglet.image import codecs
from pyglet.image.codecs import s3tc, ImageDecodeException
class DDSImageDecoder(codecs.ImageDecoder):

    def get_file_extensions(self):
        return ['.dds']

    def decode(self, filename, file):
        if not file:
            file = open(filename, 'rb')
        header = file.read(DDSURFACEDESC2.get_size())
        desc = DDSURFACEDESC2(header)
        if desc.dwMagic != b'DDS ' or desc.dwSize != 124:
            raise ImageDecodeException('Invalid DDS file (incorrect header).')
        width = desc.dwWidth
        height = desc.dwHeight
        mipmaps = 1
        if desc.dwFlags & DDSD_DEPTH:
            raise ImageDecodeException('Volume DDS files unsupported')
        if desc.dwFlags & DDSD_MIPMAPCOUNT:
            mipmaps = desc.dwMipMapCount
        if desc.ddpfPixelFormat.dwSize != 32:
            raise ImageDecodeException('Invalid DDS file (incorrect pixel format).')
        if desc.dwCaps2 & DDSCAPS2_CUBEMAP:
            raise ImageDecodeException('Cubemap DDS files unsupported')
        if not desc.ddpfPixelFormat.dwFlags & DDPF_FOURCC:
            raise ImageDecodeException('Uncompressed DDS textures not supported.')
        has_alpha = desc.ddpfPixelFormat.dwRGBAlphaBitMask != 0
        selector = (desc.ddpfPixelFormat.dwFourCC, has_alpha)
        if selector not in _compression_formats:
            raise ImageDecodeException('Unsupported texture compression %s' % desc.ddpfPixelFormat.dwFourCC)
        dformat, decoder = _compression_formats[selector]
        if dformat == GL_COMPRESSED_RGB_S3TC_DXT1_EXT:
            block_size = 8
        else:
            block_size = 16
        datas = []
        w, h = (width, height)
        for i in range(mipmaps):
            if not w and (not h):
                break
            if not w:
                w = 1
            if not h:
                h = 1
            size = (w + 3) // 4 * ((h + 3) // 4) * block_size
            data = file.read(size)
            datas.append(data)
            w >>= 1
            h >>= 1
        image = CompressedImageData(width, height, dformat, datas[0], 'GL_EXT_texture_compression_s3tc', decoder)
        level = 0
        for data in datas[1:]:
            level += 1
            image.set_mipmap_data(level, data)
        return image