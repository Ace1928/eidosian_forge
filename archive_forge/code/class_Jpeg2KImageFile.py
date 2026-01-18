from __future__ import annotations
import io
import os
import struct
from . import Image, ImageFile, _binary
class Jpeg2KImageFile(ImageFile.ImageFile):
    format = 'JPEG2000'
    format_description = 'JPEG 2000 (ISO 15444)'

    def _open(self):
        sig = self.fp.read(4)
        if sig == b'\xffO\xffQ':
            self.codec = 'j2k'
            self._size, self._mode = _parse_codestream(self.fp)
        else:
            sig = sig + self.fp.read(8)
            if sig == b'\x00\x00\x00\x0cjP  \r\n\x87\n':
                self.codec = 'jp2'
                header = _parse_jp2_header(self.fp)
                self._size, self._mode, self.custom_mimetype, dpi = header
                if dpi is not None:
                    self.info['dpi'] = dpi
                if self.fp.read(12).endswith(b'jp2c\xffO\xffQ'):
                    self._parse_comment()
            else:
                msg = 'not a JPEG 2000 file'
                raise SyntaxError(msg)
        if self.size is None or self.mode is None:
            msg = 'unable to determine size/mode'
            raise SyntaxError(msg)
        self._reduce = 0
        self.layers = 0
        fd = -1
        length = -1
        try:
            fd = self.fp.fileno()
            length = os.fstat(fd).st_size
        except Exception:
            fd = -1
            try:
                pos = self.fp.tell()
                self.fp.seek(0, io.SEEK_END)
                length = self.fp.tell()
                self.fp.seek(pos)
            except Exception:
                length = -1
        self.tile = [('jpeg2k', (0, 0) + self.size, 0, (self.codec, self._reduce, self.layers, fd, length))]

    def _parse_comment(self):
        hdr = self.fp.read(2)
        length = _binary.i16be(hdr)
        self.fp.seek(length - 2, os.SEEK_CUR)
        while True:
            marker = self.fp.read(2)
            if not marker:
                break
            typ = marker[1]
            if typ in (144, 217):
                break
            hdr = self.fp.read(2)
            length = _binary.i16be(hdr)
            if typ == 100:
                self.info['comment'] = self.fp.read(length - 2)[2:]
                break
            else:
                self.fp.seek(length - 2, os.SEEK_CUR)

    @property
    def reduce(self):
        return self._reduce or super().reduce

    @reduce.setter
    def reduce(self, value):
        self._reduce = value

    def load(self):
        if self.tile and self._reduce:
            power = 1 << self._reduce
            adjust = power >> 1
            self._size = (int((self.size[0] + adjust) / power), int((self.size[1] + adjust) / power))
            t = self.tile[0]
            t3 = (t[3][0], self._reduce, self.layers, t[3][3], t[3][4])
            self.tile = [(t[0], (0, 0) + self.size, t[2], t3)]
        return ImageFile.ImageFile.load(self)