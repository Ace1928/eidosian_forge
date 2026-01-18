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
class PDFCIDFont(PDFFont):
    default_disp: Union[float, Tuple[Optional[float], float]]

    def __init__(self, rsrcmgr: 'PDFResourceManager', spec: Mapping[str, Any], strict: bool=settings.STRICT) -> None:
        try:
            self.basefont = literal_name(spec['BaseFont'])
        except KeyError:
            if strict:
                raise PDFFontError('BaseFont is missing')
            self.basefont = 'unknown'
        self.cidsysteminfo = dict_value(spec.get('CIDSystemInfo', {}))
        cid_registry = resolve1(self.cidsysteminfo.get('Registry', b'unknown')).decode('latin1')
        cid_ordering = resolve1(self.cidsysteminfo.get('Ordering', b'unknown')).decode('latin1')
        self.cidcoding = '{}-{}'.format(cid_registry, cid_ordering)
        self.cmap: CMapBase = self.get_cmap_from_spec(spec, strict)
        try:
            descriptor = dict_value(spec['FontDescriptor'])
        except KeyError:
            if strict:
                raise PDFFontError('FontDescriptor is missing')
            descriptor = {}
        ttf = None
        if 'FontFile2' in descriptor:
            self.fontfile = stream_value(descriptor.get('FontFile2'))
            ttf = TrueTypeFont(self.basefont, BytesIO(self.fontfile.get_data()))
        self.unicode_map: Optional[UnicodeMap] = None
        if 'ToUnicode' in spec:
            if isinstance(spec['ToUnicode'], PDFStream):
                strm = stream_value(spec['ToUnicode'])
                self.unicode_map = FileUnicodeMap()
                CMapParser(self.unicode_map, BytesIO(strm.get_data())).run()
            else:
                cmap_name = literal_name(spec['ToUnicode'])
                encoding = literal_name(spec['Encoding'])
                if 'Identity' in cid_ordering or 'Identity' in cmap_name or 'Identity' in encoding:
                    self.unicode_map = IdentityUnicodeMap()
        elif self.cidcoding in ('Adobe-Identity', 'Adobe-UCS'):
            if ttf:
                try:
                    self.unicode_map = ttf.create_unicode_map()
                except TrueTypeFont.CMapNotFound:
                    pass
        else:
            try:
                self.unicode_map = CMapDB.get_unicode_map(self.cidcoding, self.cmap.is_vertical())
            except CMapDB.CMapNotFound:
                pass
        self.vertical = self.cmap.is_vertical()
        if self.vertical:
            widths2 = get_widths2(list_value(spec.get('W2', [])))
            self.disps = {cid: (vx, vy) for cid, (_, (vx, vy)) in widths2.items()}
            vy, w = resolve1(spec.get('DW2', [880, -1000]))
            self.default_disp = (None, vy)
            widths = {cid: w for cid, (w, _) in widths2.items()}
            default_width = w
        else:
            self.disps = {}
            self.default_disp = 0
            widths = get_widths(list_value(spec.get('W', [])))
            default_width = spec.get('DW', 1000)
        PDFFont.__init__(self, descriptor, widths, default_width=default_width)
        return

    def get_cmap_from_spec(self, spec: Mapping[str, Any], strict: bool) -> CMapBase:
        """Get cmap from font specification

        For certain PDFs, Encoding Type isn't mentioned as an attribute of
        Encoding but as an attribute of CMapName, where CMapName is an
        attribute of spec['Encoding'].
        The horizontal/vertical modes are mentioned with different name
        such as 'DLIdent-H/V','OneByteIdentityH/V','Identity-H/V'.
        """
        cmap_name = self._get_cmap_name(spec, strict)
        try:
            return CMapDB.get_cmap(cmap_name)
        except CMapDB.CMapNotFound as e:
            if strict:
                raise PDFFontError(e)
            return CMap()

    @staticmethod
    def _get_cmap_name(spec: Mapping[str, Any], strict: bool) -> str:
        """Get cmap name from font specification"""
        cmap_name = 'unknown'
        try:
            spec_encoding = spec['Encoding']
            if hasattr(spec_encoding, 'name'):
                cmap_name = literal_name(spec['Encoding'])
            else:
                cmap_name = literal_name(spec_encoding['CMapName'])
        except KeyError:
            if strict:
                raise PDFFontError('Encoding is unspecified')
        if type(cmap_name) is PDFStream:
            cmap_name_stream: PDFStream = cast(PDFStream, cmap_name)
            if 'CMapName' in cmap_name_stream:
                cmap_name = cmap_name_stream.get('CMapName').name
            elif strict:
                raise PDFFontError('CMapName unspecified for encoding')
        return IDENTITY_ENCODER.get(cmap_name, cmap_name)

    def __repr__(self) -> str:
        return '<PDFCIDFont: basefont={!r}, cidcoding={!r}>'.format(self.basefont, self.cidcoding)

    def is_vertical(self) -> bool:
        return self.vertical

    def is_multibyte(self) -> bool:
        return True

    def decode(self, bytes: bytes) -> Iterable[int]:
        return self.cmap.decode(bytes)

    def char_disp(self, cid: int) -> Union[float, Tuple[Optional[float], float]]:
        """Returns an integer for horizontal fonts, a tuple for vertical fonts."""
        return self.disps.get(cid, self.default_disp)

    def to_unichr(self, cid: int) -> str:
        try:
            if not self.unicode_map:
                raise KeyError(cid)
            return self.unicode_map.get_unichr(cid)
        except KeyError:
            raise PDFUnicodeNotDefined(self.cidcoding, cid)