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
class CMapBase:
    debug = 0

    def __init__(self, **kwargs: object) -> None:
        self.attrs: MutableMapping[str, object] = kwargs.copy()

    def is_vertical(self) -> bool:
        return self.attrs.get('WMode', 0) != 0

    def set_attr(self, k: str, v: object) -> None:
        self.attrs[k] = v

    def add_code2cid(self, code: str, cid: int) -> None:
        pass

    def add_cid2unichr(self, cid: int, code: Union[PSLiteral, bytes, int]) -> None:
        pass

    def use_cmap(self, cmap: 'CMapBase') -> None:
        pass

    def decode(self, code: bytes) -> Iterable[int]:
        raise NotImplementedError