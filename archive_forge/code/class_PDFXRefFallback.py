import itertools
import logging
import re
import struct
from hashlib import sha256, md5, sha384, sha512
from typing import (
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from . import settings
from .arcfour import Arcfour
from .data_structures import NumberTree
from .pdfparser import PDFSyntaxError, PDFParser, PDFStreamParser
from .pdftypes import (
from .psparser import PSEOF, literal_name, LIT, KWD
from .utils import choplist, decode_text, nunpack, format_int_roman, format_int_alpha
class PDFXRefFallback(PDFXRef):

    def __repr__(self) -> str:
        return '<PDFXRefFallback: offsets=%r>' % self.offsets.keys()
    PDFOBJ_CUE = re.compile('^(\\d+)\\s+(\\d+)\\s+obj\\b')

    def load(self, parser: PDFParser) -> None:
        parser.seek(0)
        while 1:
            try:
                pos, line_bytes = parser.nextline()
            except PSEOF:
                break
            if line_bytes.startswith(b'trailer'):
                parser.seek(pos)
                self.load_trailer(parser)
                log.debug('trailer: %r', self.trailer)
                break
            line = line_bytes.decode('latin-1')
            m = self.PDFOBJ_CUE.match(line)
            if not m:
                continue
            objid_s, genno_s = m.groups()
            objid = int(objid_s)
            genno = int(genno_s)
            self.offsets[objid] = (None, pos, genno)
            parser.seek(pos)
            _, obj = parser.nextobject()
            if isinstance(obj, PDFStream) and obj.get('Type') is LITERAL_OBJSTM:
                stream = stream_value(obj)
                try:
                    n = stream['N']
                except KeyError:
                    if settings.STRICT:
                        raise PDFSyntaxError('N is not defined: %r' % stream)
                    n = 0
                parser1 = PDFStreamParser(stream.get_data())
                objs: List[int] = []
                try:
                    while 1:
                        _, obj = parser1.nextobject()
                        objs.append(cast(int, obj))
                except PSEOF:
                    pass
                n = min(n, len(objs) // 2)
                for index in range(n):
                    objid1 = objs[index * 2]
                    self.offsets[objid1] = (objid, index, 0)