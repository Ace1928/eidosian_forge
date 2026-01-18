import ctypes
from pyglet.image import ImageData
from pyglet.image.codecs import ImageDecoder, ImageDecodeException
class BMPImageDecoder(ImageDecoder):

    def get_file_extensions(self):
        return ['.bmp']

    def decode(self, filename, file):
        if not file:
            file = open(filename, 'rb')
        bytes = file.read()
        buffer = ctypes.c_buffer(bytes)
        if bytes[:2] != b'BM':
            raise ImageDecodeException('Not a Windows bitmap file: %r' % (filename or file))
        file_header = to_ctypes(buffer, 0, BITMAPFILEHEADER)
        bits_offset = file_header.bfOffBits
        info_header_offset = ctypes.sizeof(BITMAPFILEHEADER)
        info_header = to_ctypes(buffer, info_header_offset, BITMAPINFOHEADER)
        palette_offset = info_header_offset + info_header.biSize
        if info_header.biSize < ctypes.sizeof(BITMAPINFOHEADER):
            raise ImageDecodeException('Unsupported BMP type: %r' % (filename or file))
        width = info_header.biWidth
        height = info_header.biHeight
        if width <= 0 or info_header.biPlanes != 1:
            raise ImageDecodeException('BMP file has corrupt parameters: %r' % (filename or file))
        pitch_sign = height < 0 and -1 or 1
        height = abs(height)
        compression = info_header.biCompression
        if compression not in (BI_RGB, BI_BITFIELDS):
            raise ImageDecodeException('Unsupported compression: %r' % (filename or file))
        clr_used = 0
        bitcount = info_header.biBitCount
        if bitcount == 1:
            pitch = (width + 7) // 8
            bits_type = ctypes.c_ubyte
            decoder = decode_1bit
        elif bitcount == 4:
            pitch = (width + 1) // 2
            bits_type = ctypes.c_ubyte
            decoder = decode_4bit
        elif bitcount == 8:
            bits_type = ctypes.c_ubyte
            pitch = width
            decoder = decode_8bit
        elif bitcount == 16:
            pitch = width * 2
            bits_type = ctypes.c_uint16
            decoder = decode_bitfields
        elif bitcount == 24:
            pitch = width * 3
            bits_type = ctypes.c_ubyte
            decoder = decode_24bit
        elif bitcount == 32:
            pitch = width * 4
            if compression == BI_RGB:
                decoder = decode_32bit_rgb
                bits_type = ctypes.c_ubyte
            elif compression == BI_BITFIELDS:
                decoder = decode_bitfields
                bits_type = ctypes.c_uint32
            else:
                raise ImageDecodeException('Unsupported compression: %r' % (filename or file))
        else:
            raise ImageDecodeException('Unsupported bit count %d: %r' % (bitcount, filename or file))
        pitch = pitch + 3 & ~3
        packed_width = pitch // ctypes.sizeof(bits_type)
        if bitcount < 16 and compression == BI_RGB:
            clr_used = info_header.biClrUsed or 1 << bitcount
            palette = to_ctypes(buffer, palette_offset, RGBQUAD * clr_used)
            bits = to_ctypes(buffer, bits_offset, bits_type * packed_width * height)
            return decoder(bits, palette, width, height, pitch, pitch_sign)
        elif bitcount >= 16 and compression == BI_RGB:
            bits = to_ctypes(buffer, bits_offset, bits_type * (packed_width * height))
            return decoder(bits, None, width, height, pitch, pitch_sign)
        elif compression == BI_BITFIELDS:
            if info_header.biSize >= ctypes.sizeof(BITMAPV4HEADER):
                info_header = to_ctypes(buffer, info_header_offset, BITMAPV4HEADER)
                r_mask = info_header.bV4RedMask
                g_mask = info_header.bV4GreenMask
                b_mask = info_header.bV4BlueMask
            else:
                fields_offset = info_header_offset + ctypes.sizeof(BITMAPINFOHEADER)
                fields = to_ctypes(buffer, fields_offset, RGBFields)
                r_mask = fields.red
                g_mask = fields.green
                b_mask = fields.blue

            class _BitsArray(ctypes.LittleEndianStructure):
                _pack_ = 1
                _fields_ = [('data', bits_type * packed_width * height)]
            bits = to_ctypes(buffer, bits_offset, _BitsArray).data
            return decoder(bits, r_mask, g_mask, b_mask, width, height, pitch, pitch_sign)