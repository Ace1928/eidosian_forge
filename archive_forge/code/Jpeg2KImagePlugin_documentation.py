from __future__ import annotations
import io
import os
import struct
from . import Image, ImageFile, _binary
Parse the JP2 header box to extract size, component count,
    color space information, and optionally DPI information,
    returning a (size, mode, mimetype, dpi) tuple.