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
def _read_standard_xref_table(self, stream: StreamType) -> None:
    ref = stream.read(3)
    if ref != b'ref':
        raise PdfReadError('xref table read error')
    read_non_whitespace(stream)
    stream.seek(-1, 1)
    first_time = True
    while True:
        num = cast(int, read_object(stream, self))
        if first_time and num != 0:
            self.xref_index = num
            if self.strict:
                logger_warning('Xref table not zero-indexed. ID numbers for objects will be corrected.', __name__)
        first_time = False
        read_non_whitespace(stream)
        stream.seek(-1, 1)
        size = cast(int, read_object(stream, self))
        if not isinstance(size, int):
            logger_warning('Invalid/Truncated xref table. Rebuilding it.', __name__)
            self._rebuild_xref_table(stream)
            stream.read()
            return
        read_non_whitespace(stream)
        stream.seek(-1, 1)
        cnt = 0
        while cnt < size:
            line = stream.read(20)
            while line[0] in b'\r\n':
                stream.seek(-20 + 1, 1)
                line = stream.read(20)
            if line[-1] in b'0123456789t':
                stream.seek(-1, 1)
            try:
                offset_b, generation_b = line[:16].split(b' ')
                entry_type_b = line[17:18]
                offset, generation = (int(offset_b), int(generation_b))
            except Exception:
                if hasattr(stream, 'getbuffer'):
                    buf = bytes(stream.getbuffer())
                else:
                    p = stream.tell()
                    stream.seek(0, 0)
                    buf = stream.read(-1)
                    stream.seek(p)
                f = re.search(f'{num}\\s+(\\d+)\\s+obj'.encode(), buf)
                if f is None:
                    logger_warning(f'entry {num} in Xref table invalid; object not found', __name__)
                    generation = 65535
                    offset = -1
                else:
                    logger_warning(f'entry {num} in Xref table invalid but object found', __name__)
                    generation = int(f.group(1))
                    offset = f.start()
            if generation not in self.xref:
                self.xref[generation] = {}
                self.xref_free_entry[generation] = {}
            if num in self.xref[generation]:
                pass
            else:
                if entry_type_b == b'n':
                    self.xref[generation][num] = offset
                try:
                    self.xref_free_entry[generation][num] = entry_type_b == b'f'
                except Exception:
                    pass
                try:
                    self.xref_free_entry[65535][num] = entry_type_b == b'f'
                except Exception:
                    pass
            cnt += 1
            num += 1
        read_non_whitespace(stream)
        stream.seek(-1, 1)
        trailer_tag = stream.read(7)
        if trailer_tag != b'trailer':
            stream.seek(-7, 1)
        else:
            break