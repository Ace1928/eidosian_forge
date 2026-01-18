from __future__ import absolute_import
import re
import sys
class BytesLiteralBuilder(object):
    """Assemble a byte string or char value.
    """

    def __init__(self, target_encoding):
        self.chars = []
        self.target_encoding = target_encoding

    def append(self, characters):
        if isinstance(characters, _unicode):
            characters = characters.encode(self.target_encoding)
        assert isinstance(characters, _bytes), str(type(characters))
        self.chars.append(characters)

    def append_charval(self, char_number):
        self.chars.append(_unichr(char_number).encode('ISO-8859-1'))

    def append_uescape(self, char_number, escape_string):
        self.append(escape_string)

    def getstring(self):
        return bytes_literal(join_bytes(self.chars), self.target_encoding)

    def getchar(self):
        return self.getstring()

    def getstrings(self):
        return (self.getstring(), None)