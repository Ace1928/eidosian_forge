import logging
import os
import struct
import zlib
from typing import TYPE_CHECKING, Optional, Tuple
import wandb
def _write_header(self):
    data = struct.pack('<4sHB', strtobytes(LEVELDBLOG_HEADER_IDENT), LEVELDBLOG_HEADER_MAGIC, LEVELDBLOG_HEADER_VERSION)
    assert len(data) == LEVELDBLOG_HEADER_LEN, f'header size is {len(data)} bytes, expected {LEVELDBLOG_HEADER_LEN}'
    self._fp.write(data)
    self._index += len(data)