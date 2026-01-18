import os
import zlib
import logging
from io import BytesIO
import numpy as np
from ..core import Format, read_n_bytes, image_as_uint
def _fp_read(self, n):
    return read_n_bytes(self._fp, n)