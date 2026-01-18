from __future__ import annotations
import base64
import os
import sys
import warnings
from enum import IntEnum
from io import BytesIO
from pathlib import Path
from typing import BinaryIO
from . import Image
from ._util import is_directory, is_path
def _load_pilfont(self, filename):
    with open(filename, 'rb') as fp:
        image = None
        for ext in ('.png', '.gif', '.pbm'):
            if image:
                image.close()
            try:
                fullname = os.path.splitext(filename)[0] + ext
                image = Image.open(fullname)
            except Exception:
                pass
            else:
                if image and image.mode in ('1', 'L'):
                    break
        else:
            if image:
                image.close()
            msg = 'cannot find glyph data file'
            raise OSError(msg)
        self.file = fullname
        self._load_pilfont_data(fp, image)
        image.close()