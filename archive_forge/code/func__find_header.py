import os
import json
import struct
import logging
import numpy as np
from ..core import Format
from ..v2 import imread
def _find_header(self):
    """
            Checks if file has correct header and skip it.
            """
    file_header = b'\x89LFP\r\n\x1a\n\x00\x00\x00\x01'
    header = self._file.read(HEADER_LENGTH)
    if header != file_header:
        raise RuntimeError('The LFP file header is invalid.')
    self._file.read(SIZE_LENGTH)