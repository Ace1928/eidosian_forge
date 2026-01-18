import logging
import re
import sys
from io import BytesIO
from typing import (
from .._protocols import PdfReaderProtocol, PdfWriterProtocol, XmpInformationProtocol
from .._utils import (
from ..constants import (
from ..constants import FilterTypes as FT
from ..constants import StreamAttributes as SA
from ..constants import TypArguments as TA
from ..constants import TypFitArguments as TF
from ..errors import STREAM_TRUNCATED_PREMATURELY, PdfReadError, PdfStreamError
from ._base import (
from ._fit import Fit
from ._utils import read_hex_string_from_stream, read_string_from_stream
def _to_lst(self, lst: Any) -> List[Any]:
    if isinstance(lst, (list, tuple, set)):
        pass
    elif isinstance(lst, PdfObject):
        lst = [lst]
    elif isinstance(lst, str):
        if lst[0] == '/':
            lst = [NameObject(lst)]
        else:
            lst = [TextStringObject(lst)]
    elif isinstance(lst, bytes):
        lst = [ByteStringObject(lst)]
    else:
        lst = [lst]
    return lst