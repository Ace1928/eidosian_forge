from __future__ import annotations
import io
import os
import struct
import sys
from . import Image, ImageFile, PngImagePlugin, features

        Get an icon resource as {channel: array}.  Note that
        the arrays are bottom-up like windows bitmaps and will likely
        need to be flipped or transposed in some way.
        