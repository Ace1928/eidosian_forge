from enum import IntFlag, auto
from typing import Dict, Tuple
from ._utils import deprecate_with_replacement
class FilterTypes:
    """
    Table 4.3 of the 1.4 Manual.

    Page 354 of the 1.7 Manual
    """
    ASCII_HEX_DECODE = '/ASCIIHexDecode'
    ASCII_85_DECODE = '/ASCII85Decode'
    LZW_DECODE = '/LZWDecode'
    FLATE_DECODE = '/FlateDecode'
    RUN_LENGTH_DECODE = '/RunLengthDecode'
    CCITT_FAX_DECODE = '/CCITTFaxDecode'
    DCT_DECODE = '/DCTDecode'
    JPX_DECODE = '/JPXDecode'