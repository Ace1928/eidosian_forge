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
class ASCIIHexDecode:
    """
    The ASCIIHexDecode filter decodes data that has been encoded in ASCII
    hexadecimal form into a base-7 ASCII format.
    """

    @staticmethod
    def decode(data: Union[str, bytes], decode_parms: Optional[DictionaryObject]=None, **kwargs: Any) -> bytes:
        """
        Decode an ASCII-Hex encoded data stream.

        Args:
          data: a str sequence of hexadecimal-encoded values to be
            converted into a base-7 ASCII string
          decode_parms: a string conversion in base-7 ASCII, where each of its values
            v is such that 0 <= ord(v) <= 127.

        Returns:
          A string conversion in base-7 ASCII, where each of its values
          v is such that 0 <= ord(v) <= 127.

        Raises:
          PdfStreamError:
        """
        if isinstance(data, str):
            data = data.encode()
        retval = b''
        hex_pair = b''
        index = 0
        while True:
            if index >= len(data):
                logger_warning('missing EOD in ASCIIHexDecode, check if output is OK', __name__)
                break
            char = data[index:index + 1]
            if char == b'>':
                break
            elif char.isspace():
                index += 1
                continue
            hex_pair += char
            if len(hex_pair) == 2:
                retval += bytes((int(hex_pair, base=16),))
                hex_pair = b''
            index += 1
        assert hex_pair == b''
        return retval