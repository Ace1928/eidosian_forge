from fontTools.voltLib.error import VoltLibError
from typing import NamedTuple
class GlyphDefinition(Statement):

    def __init__(self, name, gid, gunicode, gtype, components, location=None):
        Statement.__init__(self, location)
        self.name = name
        self.id = gid
        self.unicode = gunicode
        self.type = gtype
        self.components = components

    def __str__(self):
        res = f'DEF_GLYPH "{self.name}" ID {self.id}'
        if self.unicode is not None:
            if len(self.unicode) > 1:
                unicodes = ','.join((f'U+{u:04X}' for u in self.unicode))
                res += f' UNICODEVALUES "{unicodes}"'
            else:
                res += f' UNICODE {self.unicode[0]}'
        if self.type is not None:
            res += f' TYPE {self.type}'
        if self.components is not None:
            res += f' COMPONENTS {self.components}'
        res += ' END_GLYPH'
        return res