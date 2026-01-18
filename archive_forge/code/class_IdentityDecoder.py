from __future__ import annotations
import codecs
import io
import typing
import zlib
from ._compat import brotli
from ._exceptions import DecodingError
class IdentityDecoder(ContentDecoder):
    """
    Handle unencoded data.
    """

    def decode(self, data: bytes) -> bytes:
        return data

    def flush(self) -> bytes:
        return b''