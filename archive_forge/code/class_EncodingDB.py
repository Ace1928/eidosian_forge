import logging
import re
from typing import Dict, Iterable, Optional, cast
from .glyphlist import glyphname2unicode
from .latin_enc import ENCODING
from .psparser import PSLiteral
class EncodingDB:
    std2unicode: Dict[int, str] = {}
    mac2unicode: Dict[int, str] = {}
    win2unicode: Dict[int, str] = {}
    pdf2unicode: Dict[int, str] = {}
    for name, std, mac, win, pdf in ENCODING:
        c = name2unicode(name)
        if std:
            std2unicode[std] = c
        if mac:
            mac2unicode[mac] = c
        if win:
            win2unicode[win] = c
        if pdf:
            pdf2unicode[pdf] = c
    encodings = {'StandardEncoding': std2unicode, 'MacRomanEncoding': mac2unicode, 'WinAnsiEncoding': win2unicode, 'PDFDocEncoding': pdf2unicode}

    @classmethod
    def get_encoding(cls, name: str, diff: Optional[Iterable[object]]=None) -> Dict[int, str]:
        cid2unicode = cls.encodings.get(name, cls.std2unicode)
        if diff:
            cid2unicode = cid2unicode.copy()
            cid = 0
            for x in diff:
                if isinstance(x, int):
                    cid = x
                elif isinstance(x, PSLiteral):
                    try:
                        cid2unicode[cid] = name2unicode(cast(str, x.name))
                    except (KeyError, ValueError) as e:
                        log.debug(str(e))
                    cid += 1
        return cid2unicode