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
def _remove_node_from_tree(self, prev: Any, prev_ref: Any, cur: Any, last: Any) -> None:
    """
        Adjust the pointers of the linked list and tree node count.

        Args:
            prev:
            prev_ref:
            cur:
            last:
        """
    next_ref = cur.get(NameObject('/Next'), None)
    if prev is None:
        if next_ref:
            next_obj = next_ref.get_object()
            del next_obj[NameObject('/Prev')]
            self[NameObject('/First')] = next_ref
            self[NameObject('/Count')] = NumberObject(self[NameObject('/Count')] - 1)
        else:
            self[NameObject('/Count')] = NumberObject(0)
            del self[NameObject('/First')]
            if NameObject('/Last') in self:
                del self[NameObject('/Last')]
    else:
        if next_ref:
            next_obj = next_ref.get_object()
            next_obj[NameObject('/Prev')] = prev_ref
            prev[NameObject('/Next')] = next_ref
        else:
            assert cur == last
            del prev[NameObject('/Next')]
            self[NameObject('/Last')] = prev_ref
        self[NameObject('/Count')] = NumberObject(self[NameObject('/Count')] - 1)