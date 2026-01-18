from __future__ import annotations
import io
import os
import struct
from . import Image, ImageFile, _binary
def _can_read(self, num_bytes):
    if self.has_length and self.fp.tell() + num_bytes > self.length:
        return False
    if self.remaining_in_box >= 0:
        return num_bytes <= self.remaining_in_box
    else:
        return True