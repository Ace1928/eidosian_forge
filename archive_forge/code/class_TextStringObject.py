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
class TextStringObject(str, PdfObject):
    """
    A string object that has been decoded into a real unicode string.

    If read from a PDF document, this string appeared to match the
    PDFDocEncoding, or contained a UTF-16BE BOM mark to cause UTF-16 decoding
    to occur.
    """

    def clone(self, pdf_dest: Any, force_duplicate: bool=False, ignore_fields: Optional[Sequence[Union[str, int]]]=()) -> 'TextStringObject':
        """Clone object into pdf_dest."""
        obj = TextStringObject(self)
        obj.autodetect_pdfdocencoding = self.autodetect_pdfdocencoding
        obj.autodetect_utf16 = self.autodetect_utf16
        return cast('TextStringObject', self._reference_clone(obj, pdf_dest, force_duplicate))
    autodetect_pdfdocencoding = False
    autodetect_utf16 = False

    @property
    def original_bytes(self) -> bytes:
        """
        It is occasionally possible that a text string object gets created where
        a byte string object was expected due to the autodetection mechanism --
        if that occurs, this "original_bytes" property can be used to
        back-calculate what the original encoded bytes were.
        """
        return self.get_original_bytes()

    def get_original_bytes(self) -> bytes:
        if self.autodetect_utf16:
            return codecs.BOM_UTF16_BE + self.encode('utf-16be')
        elif self.autodetect_pdfdocencoding:
            return encode_pdfdocencoding(self)
        else:
            raise Exception('no information about original bytes')

    def get_encoded_bytes(self) -> bytes:
        try:
            bytearr = encode_pdfdocencoding(self)
        except UnicodeEncodeError:
            bytearr = codecs.BOM_UTF16_BE + self.encode('utf-16be')
        return bytearr

    def write_to_stream(self, stream: StreamType, encryption_key: Union[None, str, bytes]=None) -> None:
        if encryption_key is not None:
            deprecate_no_replacement('the encryption_key parameter of write_to_stream', '5.0.0')
        bytearr = self.get_encoded_bytes()
        stream.write(b'(')
        for c in bytearr:
            if not chr(c).isalnum() and c != b' ':
                stream.write(('\\%03o' % c).encode())
            else:
                stream.write(b_(chr(c)))
        stream.write(b')')