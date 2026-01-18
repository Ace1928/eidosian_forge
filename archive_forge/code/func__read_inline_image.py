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
def _read_inline_image(self, stream: StreamType) -> Dict[str, Any]:
    settings = DictionaryObject()
    while True:
        tok = read_non_whitespace(stream)
        stream.seek(-1, 1)
        if tok == b'I':
            break
        key = read_object(stream, self.pdf)
        tok = read_non_whitespace(stream)
        stream.seek(-1, 1)
        value = read_object(stream, self.pdf)
        settings[key] = value
    tmp = stream.read(3)
    assert tmp[:2] == b'ID'
    data = BytesIO()
    while True:
        buf = stream.read(8192)
        if not buf:
            raise PdfReadError('Unexpected end of stream')
        loc = buf.find(b'E')
        if loc == -1:
            data.write(buf)
        else:
            data.write(buf[0:loc])
            stream.seek(loc - len(buf), 1)
            tok = stream.read(1)
            tok2 = stream.read(1)
            if tok2 != b'I':
                stream.seek(-1, 1)
                data.write(tok)
                continue
            info = tok + tok2
            tok3 = stream.read(1)
            if tok3 not in WHITESPACES:
                stream.seek(-2, 1)
                data.write(tok)
            elif buf[loc - 1:loc] in WHITESPACES:
                while tok3 in WHITESPACES:
                    tok3 = stream.read(1)
                stream.seek(-1, 1)
                break
            else:
                while tok3 in WHITESPACES:
                    info += tok3
                    tok3 = stream.read(1)
                stream.seek(-1, 1)
                if tok3 == b'Q':
                    break
                elif tok3 == b'E':
                    ope = stream.read(3)
                    stream.seek(-3, 1)
                    if ope == b'EMC':
                        break
                else:
                    data.write(info)
    return {'settings': settings, 'data': data.getvalue()}