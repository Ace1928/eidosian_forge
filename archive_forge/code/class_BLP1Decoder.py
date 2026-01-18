from __future__ import annotations
import os
import struct
from enum import IntEnum
from io import BytesIO
from . import Image, ImageFile
class BLP1Decoder(_BLPBaseDecoder):

    def _load(self):
        if self._blp_compression == Format.JPEG:
            self._decode_jpeg_stream()
        elif self._blp_compression == 1:
            if self._blp_encoding in (4, 5):
                palette = self._read_palette()
                data = self._read_bgra(palette)
                self.set_as_raw(bytes(data))
            else:
                msg = f'Unsupported BLP encoding {repr(self._blp_encoding)}'
                raise BLPFormatError(msg)
        else:
            msg = f'Unsupported BLP compression {repr(self._blp_encoding)}'
            raise BLPFormatError(msg)

    def _decode_jpeg_stream(self):
        from .JpegImagePlugin import JpegImageFile
        jpeg_header_size, = struct.unpack('<I', self._safe_read(4))
        jpeg_header = self._safe_read(jpeg_header_size)
        self._safe_read(self._blp_offsets[0] - self.fd.tell())
        data = self._safe_read(self._blp_lengths[0])
        data = jpeg_header + data
        data = BytesIO(data)
        image = JpegImageFile(data)
        Image._decompression_bomb_check(image.size)
        if image.mode == 'CMYK':
            decoder_name, extents, offset, args = image.tile[0]
            image.tile = [(decoder_name, extents, offset, (args[0], 'CMYK'))]
        r, g, b = image.convert('RGB').split()
        image = Image.merge('RGB', (b, g, r))
        self.set_as_raw(image.tobytes())