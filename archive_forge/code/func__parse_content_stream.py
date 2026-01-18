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
def _parse_content_stream(self, stream: StreamType) -> None:
    stream.seek(0, 0)
    operands: List[Union[int, str, PdfObject]] = []
    while True:
        peek = read_non_whitespace(stream)
        if peek == b'' or peek == 0:
            break
        stream.seek(-1, 1)
        if peek.isalpha() or peek in (b"'", b'"'):
            operator = read_until_regex(stream, NameObject.delimiter_pattern)
            if operator == b'BI':
                assert operands == []
                ii = self._read_inline_image(stream)
                self._operations.append((ii, b'INLINE IMAGE'))
            else:
                self._operations.append((operands, operator))
                operands = []
        elif peek == b'%':
            while peek not in (b'\r', b'\n', b''):
                peek = stream.read(1)
        else:
            operands.append(read_object(stream, None, self.forced_encoding))