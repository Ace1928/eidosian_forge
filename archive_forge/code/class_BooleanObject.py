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
class BooleanObject(PdfObject):

    def __init__(self, value: Any) -> None:
        self.value = value

    def clone(self, pdf_dest: PdfWriterProtocol, force_duplicate: bool=False, ignore_fields: Optional[Sequence[Union[str, int]]]=()) -> 'BooleanObject':
        """Clone object into pdf_dest."""
        return cast('BooleanObject', self._reference_clone(BooleanObject(self.value), pdf_dest, force_duplicate))

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, BooleanObject):
            return self.value == __o.value
        elif isinstance(__o, bool):
            return self.value == __o
        else:
            return False

    def __repr__(self) -> str:
        return 'True' if self.value else 'False'

    def write_to_stream(self, stream: StreamType, encryption_key: Union[None, str, bytes]=None) -> None:
        if encryption_key is not None:
            deprecate_no_replacement('the encryption_key parameter of write_to_stream', '5.0.0')
        if self.value:
            stream.write(b'true')
        else:
            stream.write(b'false')

    @staticmethod
    def read_from_stream(stream: StreamType) -> 'BooleanObject':
        word = stream.read(4)
        if word == b'true':
            return BooleanObject(True)
        elif word == b'fals':
            stream.read(1)
            return BooleanObject(False)
        else:
            raise PdfReadError('Could not read Boolean object')