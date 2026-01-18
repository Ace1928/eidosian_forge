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
class UnicodeMap(CMapBase):

    def __init__(self, **kwargs: Union[str, int]) -> None:
        CMapBase.__init__(self, **kwargs)
        self.cid2unichr: Dict[int, str] = {}

    def __repr__(self) -> str:
        return '<UnicodeMap: %s>' % self.attrs.get('CMapName')

    def get_unichr(self, cid: int) -> str:
        log.debug('get_unichr: %r, %r', self, cid)
        return self.cid2unichr[cid]

    def dump(self, out: TextIO=sys.stdout) -> None:
        for k, v in sorted(self.cid2unichr.items()):
            out.write('cid %d = unicode %r\n' % (k, v))