from __future__ import annotations
from typing import BinaryIO
from . import FontFile, Image
class BdfFontFile(FontFile.FontFile):
    """Font file plugin for the X11 BDF format."""

    def __init__(self, fp: BinaryIO):
        super().__init__()
        s = fp.readline()
        if s[:13] != b'STARTFONT 2.1':
            msg = 'not a valid BDF file'
            raise SyntaxError(msg)
        props = {}
        comments = []
        while True:
            s = fp.readline()
            if not s or s[:13] == b'ENDPROPERTIES':
                break
            i = s.find(b' ')
            props[s[:i].decode('ascii')] = s[i + 1:-1].decode('ascii')
            if s[:i] in [b'COMMENT', b'COPYRIGHT']:
                if s.find(b'LogicalFontDescription') < 0:
                    comments.append(s[i + 1:-1].decode('ascii'))
        while True:
            c = bdf_char(fp)
            if not c:
                break
            id, ch, (xy, dst, src), im = c
            if 0 <= ch < len(self.glyph):
                self.glyph[ch] = (xy, dst, src, im)