from __future__ import annotations
from . import Image, ImageFile
class HDF5StubImageFile(ImageFile.StubImageFile):
    format = 'HDF5'
    format_description = 'HDF5'

    def _open(self):
        offset = self.fp.tell()
        if not _accept(self.fp.read(8)):
            msg = 'Not an HDF file'
            raise SyntaxError(msg)
        self.fp.seek(offset)
        self._mode = 'F'
        self._size = (1, 1)
        loader = self._load()
        if loader:
            loader.open(self)

    def _load(self):
        return _handler