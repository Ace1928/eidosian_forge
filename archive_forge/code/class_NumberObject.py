import binascii
import codecs
import hashlib
import re
from binascii import unhexlify
from math import log10
from typing import Any, Callable, ClassVar, Dict, Optional, Sequence, Union, cast
from .._codecs import _pdfdoc_encoding_rev
from .._protocols import PdfObjectProtocol, PdfWriterProtocol
from .._utils import (
from ..errors import STREAM_TRUNCATED_PREMATURELY, PdfReadError, PdfStreamError
class NumberObject(int, PdfObject):
    NumberPattern = re.compile(b'[^+-.0-9]')

    def __new__(cls, value: Any) -> 'NumberObject':
        try:
            return int.__new__(cls, int(value))
        except ValueError:
            logger_warning(f'NumberObject({value}) invalid; use 0 instead', __name__)
            return int.__new__(cls, 0)

    def clone(self, pdf_dest: Any, force_duplicate: bool=False, ignore_fields: Optional[Sequence[Union[str, int]]]=()) -> 'NumberObject':
        """Clone object into pdf_dest."""
        return cast('NumberObject', self._reference_clone(NumberObject(self), pdf_dest, force_duplicate))

    def as_numeric(self) -> int:
        return int(repr(self).encode('utf8'))

    def write_to_stream(self, stream: StreamType, encryption_key: Union[None, str, bytes]=None) -> None:
        if encryption_key is not None:
            deprecate_no_replacement('the encryption_key parameter of write_to_stream', '5.0.0')
        stream.write(repr(self).encode('utf8'))

    @staticmethod
    def read_from_stream(stream: StreamType) -> Union['NumberObject', 'FloatObject']:
        num = read_until_regex(stream, NumberObject.NumberPattern)
        if num.find(b'.') != -1:
            return FloatObject(num)
        return NumberObject(num)