from __future__ import annotations
from . import Image, ImageFile
from ._binary import i32le as i32
class WalImageFile(ImageFile.ImageFile):
    format = 'WAL'
    format_description = 'Quake2 Texture'

    def _open(self):
        self._mode = 'P'
        header = self.fp.read(32 + 24 + 32 + 12)
        self._size = (i32(header, 32), i32(header, 36))
        Image._decompression_bomb_check(self.size)
        offset = i32(header, 40)
        self.fp.seek(offset)
        self.info['name'] = header[:32].split(b'\x00', 1)[0]
        next_name = header[56:56 + 32].split(b'\x00', 1)[0]
        if next_name:
            self.info['next_name'] = next_name

    def load(self):
        if not self.im:
            self.im = Image.core.new(self.mode, self.size)
            self.frombytes(self.fp.read(self.size[0] * self.size[1]))
            self.putpalette(quake2palette)
        return Image.Image.load(self)