import os
import re
from io import BytesIO, UnsupportedOperation
from pathlib import Path
from typing import (
from ._doc_common import PdfDocCommon, convert_to_int
from ._encryption import Encryption, PasswordType
from ._page import PageObject
from ._utils import (
from .constants import TrailerKeys as TK
from .errors import (
from .generic import (
from .xmp import XmpInformation
def _find_startxref_pos(self, stream: StreamType) -> int:
    """
        Find startxref entry - the location of the xref table.

        Args:
            stream:

        Returns:
            The bytes offset
        """
    line = read_previous_line(stream)
    try:
        startxref = int(line)
    except ValueError:
        if not line.startswith(b'startxref'):
            raise PdfReadError('startxref not found')
        startxref = int(line[9:].strip())
        logger_warning('startxref on same line as offset', __name__)
    else:
        line = read_previous_line(stream)
        if line[:9] != b'startxref':
            raise PdfReadError('startxref not found')
    return startxref