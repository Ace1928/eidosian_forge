from __future__ import annotations
import os
from . import Image, ImageFile, ImagePalette
from ._binary import i16le as i16
from ._binary import i32le as i32
from ._binary import o8
from ._binary import o16le as o16
from ._binary import o32le as o32
class BmpRleDecoder(ImageFile.PyDecoder):
    _pulls_fd = True

    def decode(self, buffer):
        rle4 = self.args[1]
        data = bytearray()
        x = 0
        while len(data) < self.state.xsize * self.state.ysize:
            pixels = self.fd.read(1)
            byte = self.fd.read(1)
            if not pixels or not byte:
                break
            num_pixels = pixels[0]
            if num_pixels:
                if x + num_pixels > self.state.xsize:
                    num_pixels = max(0, self.state.xsize - x)
                if rle4:
                    first_pixel = o8(byte[0] >> 4)
                    second_pixel = o8(byte[0] & 15)
                    for index in range(num_pixels):
                        if index % 2 == 0:
                            data += first_pixel
                        else:
                            data += second_pixel
                else:
                    data += byte * num_pixels
                x += num_pixels
            elif byte[0] == 0:
                while len(data) % self.state.xsize != 0:
                    data += b'\x00'
                x = 0
            elif byte[0] == 1:
                break
            elif byte[0] == 2:
                bytes_read = self.fd.read(2)
                if len(bytes_read) < 2:
                    break
                right, up = self.fd.read(2)
                data += b'\x00' * (right + up * self.state.xsize)
                x = len(data) % self.state.xsize
            else:
                if rle4:
                    byte_count = byte[0] // 2
                    bytes_read = self.fd.read(byte_count)
                    for byte_read in bytes_read:
                        data += o8(byte_read >> 4)
                        data += o8(byte_read & 15)
                else:
                    byte_count = byte[0]
                    bytes_read = self.fd.read(byte_count)
                    data += bytes_read
                if len(bytes_read) < byte_count:
                    break
                x += byte[0]
                if self.fd.tell() % 2 != 0:
                    self.fd.seek(1, os.SEEK_CUR)
        rawmode = 'L' if self.mode == 'L' else 'P'
        self.set_as_raw(bytes(data), (rawmode, 0, self.args[-1]))
        return (-1, 0)