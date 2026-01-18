from __future__ import annotations
from io import BytesIO
from typing import Sequence
from . import Image, ImageFile
from ._binary import i16be as i16
from ._binary import i32be as i32
from ._deprecate import deprecate

    Get IPTC information from TIFF, JPEG, or IPTC file.

    :param im: An image containing IPTC data.
    :returns: A dictionary containing IPTC information, or None if
        no IPTC information block was found.
    