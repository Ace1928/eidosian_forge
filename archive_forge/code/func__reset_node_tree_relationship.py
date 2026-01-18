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
def _reset_node_tree_relationship(child_obj: Any) -> None:
    """
    Call this after a node has been removed from a tree.

    This resets the nodes attributes in respect to that tree.

    Args:
        child_obj:
    """
    del child_obj[NameObject('/Parent')]
    if NameObject('/Next') in child_obj:
        del child_obj[NameObject('/Next')]
    if NameObject('/Prev') in child_obj:
        del child_obj[NameObject('/Prev')]