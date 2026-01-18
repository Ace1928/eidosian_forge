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
def _rebuild_xref_table(self, stream: StreamType) -> None:
    self.xref = {}
    stream.seek(0, 0)
    f_ = stream.read(-1)
    for m in re.finditer(b'[\\r\\n \\t][ \\t]*(\\d+)[ \\t]+(\\d+)[ \\t]+obj', f_):
        idnum = int(m.group(1))
        generation = int(m.group(2))
        if generation not in self.xref:
            self.xref[generation] = {}
        self.xref[generation][idnum] = m.start(1)
    stream.seek(0, 0)
    for m in re.finditer(b'[\\r\\n \\t][ \\t]*trailer[\\r\\n \\t]*(<<)', f_):
        stream.seek(m.start(1), 0)
        new_trailer = cast(Dict[Any, Any], read_object(stream, self))
        for key, value in list(new_trailer.items()):
            self.trailer[key] = value