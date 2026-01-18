from __future__ import annotations
import calendar
import codecs
import collections
import mmap
import os
import re
import time
import zlib
class PdfName:

    def __init__(self, name):
        if isinstance(name, PdfName):
            self.name = name.name
        elif isinstance(name, bytes):
            self.name = name
        else:
            self.name = name.encode('us-ascii')

    def name_as_str(self):
        return self.name.decode('us-ascii')

    def __eq__(self, other):
        return isinstance(other, PdfName) and other.name == self.name or other == self.name

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return f'PdfName({repr(self.name)})'

    @classmethod
    def from_pdf_stream(cls, data):
        return cls(PdfParser.interpret_name(data))
    allowed_chars = set(range(33, 127)) - {ord(c) for c in '#%/()<>[]{}'}

    def __bytes__(self):
        result = bytearray(b'/')
        for b in self.name:
            if b in self.allowed_chars:
                result.append(b)
            else:
                result.extend(b'#%02X' % b)
        return bytes(result)