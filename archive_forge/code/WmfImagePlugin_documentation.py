from __future__ import annotations
from . import Image, ImageFile
from ._binary import i16le as word
from ._binary import si16le as short
from ._binary import si32le as _long

    Install application-specific WMF image handler.

    :param handler: Handler object.
    