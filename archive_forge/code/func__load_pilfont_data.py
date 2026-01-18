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
def _load_pilfont_data(self, file, image):
    if file.readline() != b'PILfont\n':
        msg = 'Not a PILfont file'
        raise SyntaxError(msg)
    file.readline().split(b';')
    self.info = []
    while True:
        s = file.readline()
        if not s or s == b'DATA\n':
            break
        self.info.append(s)
    data = file.read(256 * 20)
    if image.mode not in ('1', 'L'):
        msg = 'invalid font image mode'
        raise TypeError(msg)
    image.load()
    self.font = Image.core.font(image.im, data)