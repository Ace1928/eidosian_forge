from fontTools.misc.textTools import byteord, strjoin, tobytes, tostr
import sys
import os
import string
def _writeraw(self, data, indent=True, strip=False):
    """Writes bytes, possibly indented."""
    if indent and self.needindent:
        self.file.write(self.indentlevel * self.indentwhite)
        self.needindent = 0
    s = self.totype(data, encoding='utf_8')
    if strip:
        s = s.strip()
    self.file.write(s)