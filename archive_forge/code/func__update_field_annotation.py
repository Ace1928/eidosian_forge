import codecs
import collections
import decimal
import enum
import hashlib
import re
import uuid
from io import BytesIO, FileIO, IOBase
from pathlib import Path
from types import TracebackType
from typing import (
from ._cmap import build_char_map_from_dict
from ._doc_common import PdfDocCommon
from ._encryption import EncryptAlgorithm, Encryption
from ._page import PageObject
from ._page_labels import nums_clear_range, nums_insert, nums_next
from ._reader import PdfReader
from ._utils import (
from .constants import AnnotationDictionaryAttributes as AA
from .constants import CatalogAttributes as CA
from .constants import (
from .constants import CatalogDictionary as CD
from .constants import Core as CO
from .constants import (
from .constants import PageAttributes as PG
from .constants import PagesAttributes as PA
from .constants import TrailerKeys as TK
from .errors import PyPdfError
from .generic import (
from .pagerange import PageRange, PageRangeSpec
from .types import (
from .xmp import XmpInformation
def _update_field_annotation(self, field: DictionaryObject, anno: DictionaryObject) -> None:
    _rct = cast(RectangleObject, anno[AA.Rect])
    rct = RectangleObject((0, 0, _rct[2] - _rct[0], _rct[3] - _rct[1]))
    da = anno.get_inherited(AA.DA, cast(DictionaryObject, self.root_object[CatalogDictionary.ACRO_FORM]).get(AA.DA, None))
    if da is None:
        da = TextStringObject('/Helv 0 Tf 0 g')
    else:
        da = da.get_object()
    font_properties = da.replace('\n', ' ').replace('\r', ' ').split(' ')
    font_properties = [x for x in font_properties if x != '']
    font_name = font_properties[font_properties.index('Tf') - 2]
    font_height = float(font_properties[font_properties.index('Tf') - 1])
    if font_height == 0:
        font_height = rct.height - 2
        font_properties[font_properties.index('Tf') - 1] = str(font_height)
        da = ' '.join(font_properties)
    y_offset = rct.height - 1 - font_height
    dr: Any = cast(DictionaryObject, cast(DictionaryObject, anno.get_inherited('/DR', cast(DictionaryObject, self.root_object[CatalogDictionary.ACRO_FORM]).get('/DR', DictionaryObject()))).get_object())
    dr = dr.get('/Font', DictionaryObject()).get_object()
    if font_name not in dr:
        dr = cast(Dict[Any, Any], cast(DictionaryObject, self.root_object[CatalogDictionary.ACRO_FORM]).get('/DR', {}))
        dr = dr.get_object().get('/Font', DictionaryObject()).get_object()
    font_res = dr.get(font_name, None)
    if font_res is not None:
        font_res = cast(DictionaryObject, font_res.get_object())
        font_subtype, _, font_encoding, font_map = build_char_map_from_dict(200, font_res)
        try:
            del font_map[-1]
        except KeyError:
            pass
        font_full_rev: Dict[str, bytes]
        if isinstance(font_encoding, str):
            font_full_rev = {v: k.encode(font_encoding) for k, v in font_map.items()}
        else:
            font_full_rev = {v: bytes((k,)) for k, v in font_encoding.items()}
            font_encoding_rev = {v: bytes((k,)) for k, v in font_encoding.items()}
            for kk, v in font_map.items():
                font_full_rev[v] = font_encoding_rev.get(kk, kk)
    else:
        logger_warning(f'Font dictionary for {font_name} not found.', __name__)
        font_full_rev = {}
    field_flags = field.get(FA.Ff, 0)
    if field.get(FA.FT, '/Tx') == '/Ch' and field_flags & FA.FfBits.Combo == 0:
        txt = '\n'.join(anno.get_inherited(FA.Opt, []))
        sel = field.get('/V', [])
        if not isinstance(sel, list):
            sel = [sel]
    else:
        txt = field.get('/V', '')
        sel = []
    txt = txt.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')
    ap_stream = f'q\n/Tx BMC \nq\n1 1 {rct.width - 1} {rct.height - 1} re\nW\nBT\n{da}\n'.encode()
    for line_number, line in enumerate(txt.replace('\n', '\r').split('\r')):
        if line in sel:
            ap_stream += f'1 {y_offset - line_number * font_height * 1.4 - 1} {rct.width - 2} {font_height + 2} re\n0.5 0.5 0.5 rg s\n{da}\n'.encode()
        if line_number == 0:
            ap_stream += f'2 {y_offset} Td\n'.encode()
        else:
            ap_stream += f'0 {-font_height * 1.4} Td\n'.encode()
        enc_line: List[bytes] = [font_full_rev.get(c, c.encode('utf-16-be')) for c in line]
        if any((len(c) >= 2 for c in enc_line)):
            ap_stream += b'<' + b''.join(enc_line).hex().encode() + b'> Tj\n'
        else:
            ap_stream += b'(' + b''.join(enc_line) + b') Tj\n'
    ap_stream += b'ET\nQ\nEMC\nQ\n'
    dct = DecodedStreamObject.initialize_from_dictionary({NameObject('/Type'): NameObject('/XObject'), NameObject('/Subtype'): NameObject('/Form'), NameObject('/BBox'): rct, '__streamdata__': ByteStringObject(ap_stream), '/Length': 0})
    if font_res is not None:
        dct[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject(font_name): getattr(font_res, 'indirect_reference', font_res)})})
    if AA.AP not in anno:
        anno[NameObject(AA.AP)] = DictionaryObject({NameObject('/N'): self._add_object(dct)})
    elif '/N' not in cast(DictionaryObject, anno[AA.AP]):
        cast(DictionaryObject, anno[NameObject(AA.AP)])[NameObject('/N')] = self._add_object(dct)
    else:
        n = anno[AA.AP]['/N'].indirect_reference.idnum
        self._objects[n - 1] = dct
        dct.indirect_reference = IndirectObject(n, 0, self)