import logging
import re
from io import BytesIO
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union, cast
from . import settings
from .cmapdb import CMap
from .cmapdb import CMapBase
from .cmapdb import CMapDB
from .pdfcolor import PDFColorSpace
from .pdfcolor import PREDEFINED_COLORSPACE
from .pdfdevice import PDFDevice
from .pdfdevice import PDFTextSeq
from .pdffont import PDFCIDFont
from .pdffont import PDFFont
from .pdffont import PDFFontError
from .pdffont import PDFTrueTypeFont
from .pdffont import PDFType1Font
from .pdffont import PDFType3Font
from .pdfpage import PDFPage
from .pdftypes import PDFException
from .pdftypes import PDFObjRef
from .pdftypes import PDFStream
from .pdftypes import dict_value
from .pdftypes import list_value
from .pdftypes import resolve1
from .pdftypes import stream_value
from .psparser import KWD
from .psparser import LIT
from .psparser import PSEOF
from .psparser import PSKeyword
from .psparser import PSLiteral, PSTypeError
from .psparser import PSStackParser
from .psparser import PSStackType
from .psparser import keyword_name
from .psparser import literal_name
from .utils import MATRIX_IDENTITY
from .utils import Matrix, Point, PathSegment, Rect
from .utils import choplist
from .utils import mult_matrix
class PDFContentParser(PSStackParser[Union[PSKeyword, PDFStream]]):

    def __init__(self, streams: Sequence[object]) -> None:
        self.streams = streams
        self.istream = 0
        PSStackParser.__init__(self, None)

    def fillfp(self) -> None:
        if not self.fp:
            if self.istream < len(self.streams):
                strm = stream_value(self.streams[self.istream])
                self.istream += 1
            else:
                raise PSEOF('Unexpected EOF, file truncated?')
            self.fp = BytesIO(strm.get_data())

    def seek(self, pos: int) -> None:
        self.fillfp()
        PSStackParser.seek(self, pos)

    def fillbuf(self) -> None:
        if self.charpos < len(self.buf):
            return
        while 1:
            self.fillfp()
            self.bufpos = self.fp.tell()
            self.buf = self.fp.read(self.BUFSIZ)
            if self.buf:
                break
            self.fp = None
        self.charpos = 0

    def get_inline_data(self, pos: int, target: bytes=b'EI') -> Tuple[int, bytes]:
        self.seek(pos)
        i = 0
        data = b''
        while i <= len(target):
            self.fillbuf()
            if i:
                ci = self.buf[self.charpos]
                c = bytes((ci,))
                data += c
                self.charpos += 1
                if len(target) <= i and c.isspace():
                    i += 1
                elif i < len(target) and c == bytes((target[i],)):
                    i += 1
                else:
                    i = 0
            else:
                try:
                    j = self.buf.index(target[0], self.charpos)
                    data += self.buf[self.charpos:j + 1]
                    self.charpos = j + 1
                    i = 1
                except ValueError:
                    data += self.buf[self.charpos:]
                    self.charpos = len(self.buf)
        data = data[:-(len(target) + 1)]
        data = re.sub(b'(\\x0d\\x0a|[\\x0d\\x0a])$', b'', data)
        return (pos, data)

    def flush(self) -> None:
        self.add_results(*self.popall())
    KEYWORD_BI = KWD(b'BI')
    KEYWORD_ID = KWD(b'ID')
    KEYWORD_EI = KWD(b'EI')

    def do_keyword(self, pos: int, token: PSKeyword) -> None:
        if token is self.KEYWORD_BI:
            self.start_type(pos, 'inline')
        elif token is self.KEYWORD_ID:
            try:
                _, objs = self.end_type('inline')
                if len(objs) % 2 != 0:
                    error_msg = 'Invalid dictionary construct: {!r}'.format(objs)
                    raise PSTypeError(error_msg)
                d = {literal_name(k): v for k, v in choplist(2, objs)}
                pos, data = self.get_inline_data(pos + len(b'ID '))
                obj = PDFStream(d, data)
                self.push((pos, obj))
                self.push((pos, self.KEYWORD_EI))
            except PSTypeError:
                if settings.STRICT:
                    raise
        else:
            self.push((pos, token))