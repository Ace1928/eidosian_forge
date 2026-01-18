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
class FlateDecode:

    @staticmethod
    def decode(data: bytes, decode_parms: Optional[DictionaryObject]=None, **kwargs: Any) -> bytes:
        """
        Decode data which is flate-encoded.

        Args:
          data: flate-encoded data.
          decode_parms: a dictionary of values, understanding the
            "/Predictor":<int> key only

        Returns:
          The flate-decoded data.

        Raises:
          PdfReadError:
        """
        if 'decodeParms' in kwargs:
            deprecate_with_replacement('decodeParms', 'parameters', '4.0.0')
            decode_parms = kwargs['decodeParms']
        if isinstance(decode_parms, ArrayObject):
            raise DeprecationError('decode_parms as ArrayObject is depreciated')
        str_data = decompress(data)
        predictor = 1
        if decode_parms:
            try:
                predictor = decode_parms.get('/Predictor', 1)
            except (AttributeError, TypeError):
                pass
        if predictor != 1:
            DEFAULT_BITS_PER_COMPONENT = 8
            try:
                columns = cast(int, decode_parms[LZW.COLUMNS].get_object())
            except (TypeError, KeyError):
                columns = 1
            try:
                colors = cast(int, decode_parms[LZW.COLORS].get_object())
            except (TypeError, KeyError):
                colors = 1
            try:
                bits_per_component = cast(int, decode_parms[LZW.BITS_PER_COMPONENT].get_object())
            except (TypeError, KeyError):
                bits_per_component = DEFAULT_BITS_PER_COMPONENT
            rowlength = math.ceil(columns * colors * bits_per_component / 8) + 1
            if predictor == 2:
                rowlength -= 1
                bpp = rowlength // columns
                str_data = bytearray(str_data)
                for i in range(len(str_data)):
                    if i % rowlength >= bpp:
                        str_data[i] = (str_data[i] + str_data[i - bpp]) % 256
                str_data = bytes(str_data)
            elif 10 <= predictor <= 15:
                str_data = FlateDecode._decode_png_prediction(str_data, columns, rowlength)
            else:
                raise PdfReadError(f'Unsupported flatedecode predictor {predictor!r}')
        return str_data

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

    @staticmethod
    def encode(data: bytes, level: int=-1) -> bytes:
        """
        Compress the input data using zlib.

        Args:
            data: The data to be compressed.
            level: See https://docs.python.org/3/library/zlib.html#zlib.compress

        Returns:
            The compressed data.
        """
        return zlib.compress(data, level)