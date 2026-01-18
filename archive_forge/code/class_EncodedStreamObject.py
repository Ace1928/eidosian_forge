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
class EncodedStreamObject(StreamObject):

    def __init__(self) -> None:
        self.decoded_self: Optional[DecodedStreamObject] = None

    def get_data(self) -> Union[bytes, str]:
        from ..filters import decode_stream_data
        if self.decoded_self is not None:
            return self.decoded_self.get_data()
        else:
            decoded = DecodedStreamObject()
            decoded.set_data(b_(decode_stream_data(self)))
            for key, value in list(self.items()):
                if key not in (SA.LENGTH, SA.FILTER, SA.DECODE_PARMS):
                    decoded[key] = value
            self.decoded_self = decoded
            return decoded.get_data()

    def set_data(self, data: bytes) -> None:
        from ..filters import FlateDecode
        if self.get(SA.FILTER, '') == FT.FLATE_DECODE:
            if not isinstance(data, bytes):
                raise TypeError('data must be bytes')
            assert self.decoded_self is not None
            self.decoded_self.set_data(data)
            super().set_data(FlateDecode.encode(data))
        else:
            raise PdfReadError('Streams encoded with different filter from only FlateDecode is not supported')