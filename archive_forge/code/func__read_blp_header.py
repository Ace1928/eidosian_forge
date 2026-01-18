from __future__ import annotations
import os
import struct
from enum import IntEnum
from io import BytesIO
from . import Image, ImageFile
def _read_blp_header(self):
    self.fd.seek(4)
    self._blp_compression, = struct.unpack('<i', self._safe_read(4))
    self._blp_encoding, = struct.unpack('<b', self._safe_read(1))
    self._blp_alpha_depth, = struct.unpack('<b', self._safe_read(1))
    self._blp_alpha_encoding, = struct.unpack('<b', self._safe_read(1))
    self.fd.seek(1, os.SEEK_CUR)
    self.size = struct.unpack('<II', self._safe_read(8))
    if isinstance(self, BLP1Decoder):
        self._blp_encoding, = struct.unpack('<i', self._safe_read(4))
        self.fd.seek(4, os.SEEK_CUR)
    self._blp_offsets = struct.unpack('<16I', self._safe_read(16 * 4))
    self._blp_lengths = struct.unpack('<16I', self._safe_read(16 * 4))