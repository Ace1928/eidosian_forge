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
class NullObject(PdfObject):

    def clone(self, pdf_dest: PdfWriterProtocol, force_duplicate: bool=False, ignore_fields: Optional[Sequence[Union[str, int]]]=()) -> 'NullObject':
        """Clone object into pdf_dest."""
        return cast('NullObject', self._reference_clone(NullObject(), pdf_dest, force_duplicate))

    def write_to_stream(self, stream: StreamType, encryption_key: Union[None, str, bytes]=None) -> None:
        if encryption_key is not None:
            deprecate_no_replacement('the encryption_key parameter of write_to_stream', '5.0.0')
        stream.write(b'null')

    @staticmethod
    def read_from_stream(stream: StreamType) -> 'NullObject':
        nulltxt = stream.read(4)
        if nulltxt != b'null':
            raise PdfReadError('Could not read Null object')
        return NullObject()

    def __repr__(self) -> str:
        return 'NullObject'