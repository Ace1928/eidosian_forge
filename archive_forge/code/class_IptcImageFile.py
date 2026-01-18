from __future__ import annotations
from io import BytesIO
from typing import Sequence
from . import Image, ImageFile
from ._binary import i16be as i16
from ._binary import i32be as i32
from ._deprecate import deprecate
class IptcImageFile(ImageFile.ImageFile):
    format = 'IPTC'
    format_description = 'IPTC/NAA'

    def getint(self, key: tuple[int, int]) -> int:
        return _i(self.info[key])

    def field(self) -> tuple[tuple[int, int] | None, int]:
        s = self.fp.read(5)
        if not s.strip(b'\x00'):
            return (None, 0)
        tag = (s[1], s[2])
        if s[0] != 28 or tag[0] not in [1, 2, 3, 4, 5, 6, 7, 8, 9, 240]:
            msg = 'invalid IPTC/NAA file'
            raise SyntaxError(msg)
        size = s[3]
        if size > 132:
            msg = 'illegal field length in IPTC/NAA file'
            raise OSError(msg)
        elif size == 128:
            size = 0
        elif size > 128:
            size = _i(self.fp.read(size - 128))
        else:
            size = i16(s, 3)
        return (tag, size)

    def _open(self) -> None:
        while True:
            offset = self.fp.tell()
            tag, size = self.field()
            if not tag or tag == (8, 10):
                break
            if size:
                tagdata = self.fp.read(size)
            else:
                tagdata = None
            if tag in self.info:
                if isinstance(self.info[tag], list):
                    self.info[tag].append(tagdata)
                else:
                    self.info[tag] = [self.info[tag], tagdata]
            else:
                self.info[tag] = tagdata
        layers = self.info[3, 60][0]
        component = self.info[3, 60][1]
        if (3, 65) in self.info:
            id = self.info[3, 65][0] - 1
        else:
            id = 0
        if layers == 1 and (not component):
            self._mode = 'L'
        elif layers == 3 and component:
            self._mode = 'RGB'[id]
        elif layers == 4 and component:
            self._mode = 'CMYK'[id]
        self._size = (self.getint((3, 20)), self.getint((3, 30)))
        try:
            compression = COMPRESSION[self.getint((3, 120))]
        except KeyError as e:
            msg = 'Unknown IPTC image compression'
            raise OSError(msg) from e
        if tag == (8, 10):
            self.tile = [('iptc', (0, 0) + self.size, offset, compression)]

    def load(self):
        if len(self.tile) != 1 or self.tile[0][0] != 'iptc':
            return ImageFile.ImageFile.load(self)
        offset, compression = self.tile[0][2:]
        self.fp.seek(offset)
        o = BytesIO()
        if compression == 'raw':
            o.write(b'P5\n%d %d\n255\n' % self.size)
        while True:
            type, size = self.field()
            if type != (8, 10):
                break
            while size > 0:
                s = self.fp.read(min(size, 8192))
                if not s:
                    break
                o.write(s)
                size -= len(s)
        with Image.open(o) as _im:
            _im.load()
            self.im = _im.im