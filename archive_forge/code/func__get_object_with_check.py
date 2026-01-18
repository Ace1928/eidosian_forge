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
def _get_object_with_check(self) -> Optional['PdfObject']:
    o = self.get_object()
    if isinstance(o, IndirectObject):
        raise PdfStreamError(f'{self.__repr__()} references an IndirectObject {o.__repr__()}')
    return o