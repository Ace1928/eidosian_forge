import io
import logging
import sys
import zlib
from typing import (
from . import settings
from .ascii85 import ascii85decode
from .ascii85 import asciihexdecode
from .ccitt import ccittfaxdecode
from .lzw import lzwdecode
from .psparser import LIT
from .psparser import PSException
from .psparser import PSObject
from .runlength import rldecode
from .utils import apply_png_predictor
class DecipherCallable(Protocol):
    """Fully typed a decipher callback, with optional parameter."""

    def __call__(self, objid: int, genno: int, data: bytes, attrs: Optional[Dict[str, Any]]=None) -> bytes:
        raise NotImplementedError