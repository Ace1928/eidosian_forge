from enum import IntFlag, auto
from typing import Dict, Tuple
from ._utils import deprecate_with_replacement
class CcittFaxDecodeParameters:
    """Table 4.5."""
    K = '/K'
    END_OF_LINE = '/EndOfLine'
    ENCODED_BYTE_ALIGN = '/EncodedByteAlign'
    COLUMNS = '/Columns'
    ROWS = '/Rows'
    END_OF_BLOCK = '/EndOfBlock'
    BLACK_IS_1 = '/BlackIs1'
    DAMAGED_ROWS_BEFORE_ERROR = '/DamagedRowsBeforeError'