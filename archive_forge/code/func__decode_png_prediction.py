import math
import struct
import zlib
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union, cast
from ._utils import (
from .constants import CcittFaxDecodeParameters as CCITT
from .constants import ColorSpaces
from .constants import FilterTypeAbbreviations as FTA
from .constants import FilterTypes as FT
from .constants import ImageAttributes as IA
from .constants import LzwFilterParameters as LZW
from .constants import StreamAttributes as SA
from .errors import DeprecationError, PdfReadError, PdfStreamError
from .generic import (
@staticmethod
def _decode_png_prediction(data: bytes, columns: int, rowlength: int) -> bytes:
    if len(data) % rowlength != 0:
        raise PdfReadError('Image data is not rectangular')
    output = []
    prev_rowdata = (0,) * rowlength
    bpp = (rowlength - 1) // columns
    for row in range(0, len(data), rowlength):
        rowdata: List[int] = list(data[row:row + rowlength])
        filter_byte = rowdata[0]
        if filter_byte == 0:
            pass
        elif filter_byte == 1:
            for i in range(bpp + 1, rowlength):
                rowdata[i] = (rowdata[i] + rowdata[i - bpp]) % 256
        elif filter_byte == 2:
            for i in range(1, rowlength):
                rowdata[i] = (rowdata[i] + prev_rowdata[i]) % 256
        elif filter_byte == 3:
            for i in range(1, bpp + 1):
                floor = prev_rowdata[i] // 2
                rowdata[i] = (rowdata[i] + floor) % 256
            for i in range(bpp + 1, rowlength):
                left = rowdata[i - bpp]
                floor = (left + prev_rowdata[i]) // 2
                rowdata[i] = (rowdata[i] + floor) % 256
        elif filter_byte == 4:
            for i in range(1, bpp + 1):
                up = prev_rowdata[i]
                paeth = up
                rowdata[i] = (rowdata[i] + paeth) % 256
            for i in range(bpp + 1, rowlength):
                left = rowdata[i - bpp]
                up = prev_rowdata[i]
                up_left = prev_rowdata[i - bpp]
                p = left + up - up_left
                dist_left = abs(p - left)
                dist_up = abs(p - up)
                dist_up_left = abs(p - up_left)
                if dist_left <= dist_up and dist_left <= dist_up_left:
                    paeth = left
                elif dist_up <= dist_up_left:
                    paeth = up
                else:
                    paeth = up_left
                rowdata[i] = (rowdata[i] + paeth) % 256
        else:
            raise PdfReadError(f'Unsupported PNG filter {filter_byte!r}')
        prev_rowdata = tuple(rowdata)
        output.extend(rowdata[1:])
    return bytes(output)