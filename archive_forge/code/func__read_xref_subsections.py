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
def _read_xref_subsections(self, idx_pairs: List[int], get_entry: Callable[[int], Union[int, Tuple[int, ...]]], used_before: Callable[[int, Union[int, Tuple[int, ...]]], bool]) -> None:
    for start, size in self._pairs(idx_pairs):
        for num in range(start, start + size):
            xref_type = get_entry(0)
            if xref_type == 0:
                next_free_object = get_entry(1)
                next_generation = get_entry(2)
            elif xref_type == 1:
                byte_offset = get_entry(1)
                generation = get_entry(2)
                if generation not in self.xref:
                    self.xref[generation] = {}
                if not used_before(num, generation):
                    self.xref[generation][num] = byte_offset
            elif xref_type == 2:
                objstr_num = get_entry(1)
                obstr_idx = get_entry(2)
                generation = 0
                if not used_before(num, generation):
                    self.xref_objStm[num] = (objstr_num, obstr_idx)
            elif self.strict:
                raise PdfReadError(f'Unknown xref type: {xref_type}')