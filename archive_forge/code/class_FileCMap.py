import gzip
import logging
import os
import os.path
import pickle as pickle
import struct
import sys
from typing import (
from .encodingdb import name2unicode
from .psparser import KWD
from .psparser import PSEOF
from .psparser import PSKeyword
from .psparser import PSLiteral
from .psparser import PSStackParser
from .psparser import PSSyntaxError
from .psparser import literal_name
from .utils import choplist
from .utils import nunpack
class FileCMap(CMap):

    def add_code2cid(self, code: str, cid: int) -> None:
        assert isinstance(code, str) and isinstance(cid, int), str((type(code), type(cid)))
        d = self.code2cid
        for c in code[:-1]:
            ci = ord(c)
            if ci in d:
                d = cast(Dict[int, object], d[ci])
            else:
                t: Dict[int, object] = {}
                d[ci] = t
                d = t
        ci = ord(code[-1])
        d[ci] = cid