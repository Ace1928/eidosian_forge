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
def _read_xref_tables_and_trailers(self, stream: StreamType, startxref: Optional[int], xref_issue_nr: int) -> None:
    self.xref: Dict[int, Dict[Any, Any]] = {}
    self.xref_free_entry: Dict[int, Dict[Any, Any]] = {}
    self.xref_objStm: Dict[int, Tuple[Any, Any]] = {}
    self.trailer = DictionaryObject()
    while startxref is not None:
        stream.seek(startxref, 0)
        x = stream.read(1)
        if x in b'\r\n':
            x = stream.read(1)
        if x == b'x':
            startxref = self._read_xref(stream)
        elif xref_issue_nr:
            try:
                self._rebuild_xref_table(stream)
                break
            except Exception:
                xref_issue_nr = 0
        elif x.isdigit():
            try:
                xrefstream = self._read_pdf15_xref_stream(stream)
            except Exception as e:
                if TK.ROOT in self.trailer:
                    logger_warning(f'Previous trailer can not be read {e.args}', __name__)
                    break
                else:
                    raise PdfReadError(f'trailer can not be read {e.args}')
            trailer_keys = (TK.ROOT, TK.ENCRYPT, TK.INFO, TK.ID, TK.SIZE)
            for key in trailer_keys:
                if key in xrefstream and key not in self.trailer:
                    self.trailer[NameObject(key)] = xrefstream.raw_get(key)
            if '/XRefStm' in xrefstream:
                p = stream.tell()
                stream.seek(cast(int, xrefstream['/XRefStm']) + 1, 0)
                self._read_pdf15_xref_stream(stream)
                stream.seek(p, 0)
            if '/Prev' in xrefstream:
                startxref = cast(int, xrefstream['/Prev'])
            else:
                break
        else:
            startxref = self._read_xref_other_error(stream, startxref)