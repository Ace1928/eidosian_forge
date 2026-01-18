import logging
import struct
import sys
from io import BytesIO
from typing import (
from . import settings
from .cmapdb import CMap
from .cmapdb import CMapBase
from .cmapdb import CMapDB
from .cmapdb import CMapParser
from .cmapdb import FileUnicodeMap
from .cmapdb import IdentityUnicodeMap
from .cmapdb import UnicodeMap
from .encodingdb import EncodingDB
from .encodingdb import name2unicode
from .fontmetrics import FONT_METRICS
from .pdftypes import PDFException
from .pdftypes import PDFStream
from .pdftypes import dict_value
from .pdftypes import int_value
from .pdftypes import list_value
from .pdftypes import num_value
from .pdftypes import resolve1, resolve_all
from .pdftypes import stream_value
from .psparser import KWD
from .psparser import LIT
from .psparser import PSEOF
from .psparser import PSKeyword
from .psparser import PSLiteral
from .psparser import PSStackParser
from .psparser import literal_name
from .utils import Matrix, Point
from .utils import Rect
from .utils import apply_matrix_norm
from .utils import choplist
from .utils import nunpack
class PDFFont:

    def __init__(self, descriptor: Mapping[str, Any], widths: FontWidthDict, default_width: Optional[float]=None) -> None:
        self.descriptor = descriptor
        self.widths: FontWidthDict = resolve_all(widths)
        self.fontname = resolve1(descriptor.get('FontName', 'unknown'))
        if isinstance(self.fontname, PSLiteral):
            self.fontname = literal_name(self.fontname)
        self.flags = int_value(descriptor.get('Flags', 0))
        self.ascent = num_value(descriptor.get('Ascent', 0))
        self.descent = num_value(descriptor.get('Descent', 0))
        self.italic_angle = num_value(descriptor.get('ItalicAngle', 0))
        if default_width is None:
            self.default_width = num_value(descriptor.get('MissingWidth', 0))
        else:
            self.default_width = default_width
        self.default_width = resolve1(self.default_width)
        self.leading = num_value(descriptor.get('Leading', 0))
        self.bbox = cast(Rect, list_value(resolve_all(descriptor.get('FontBBox', (0, 0, 0, 0)))))
        self.hscale = self.vscale = 0.001
        if self.descent > 0:
            self.descent = -self.descent
        return

    def __repr__(self) -> str:
        return '<PDFFont>'

    def is_vertical(self) -> bool:
        return False

    def is_multibyte(self) -> bool:
        return False

    def decode(self, bytes: bytes) -> Iterable[int]:
        return bytearray(bytes)

    def get_ascent(self) -> float:
        """Ascent above the baseline, in text space units"""
        return self.ascent * self.vscale

    def get_descent(self) -> float:
        """Descent below the baseline, in text space units; always negative"""
        return self.descent * self.vscale

    def get_width(self) -> float:
        w = self.bbox[2] - self.bbox[0]
        if w == 0:
            w = -self.default_width
        return w * self.hscale

    def get_height(self) -> float:
        h = self.bbox[3] - self.bbox[1]
        if h == 0:
            h = self.ascent - self.descent
        return h * self.vscale

    def char_width(self, cid: int) -> float:
        try:
            return cast(Dict[int, float], self.widths)[cid] * self.hscale
        except KeyError:
            str_widths = cast(Dict[str, float], self.widths)
            try:
                return str_widths[self.to_unichr(cid)] * self.hscale
            except (KeyError, PDFUnicodeNotDefined):
                return self.default_width * self.hscale

    def char_disp(self, cid: int) -> Union[float, Tuple[Optional[float], float]]:
        """Returns an integer for horizontal fonts, a tuple for vertical fonts."""
        return 0

    def string_width(self, s: bytes) -> float:
        return sum((self.char_width(cid) for cid in self.decode(s)))

    def to_unichr(self, cid: int) -> str:
        raise NotImplementedError