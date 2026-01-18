from __future__ import annotations
from . import Image, ImageFile
class BufrStubImageFile(ImageFile.StubImageFile):
    format = 'BUFR'
    format_description = 'BUFR'

    def _open(self):
        offset = self.fp.tell()
        if not _accept(self.fp.read(4)):
            msg = 'Not a BUFR file'
            raise SyntaxError(msg)
        self.fp.seek(offset)
        self._mode = 'F'
        self._size = (1, 1)
        loader = self._load()
        if loader:
            loader.open(self)

    def _load(self):
        return _handler