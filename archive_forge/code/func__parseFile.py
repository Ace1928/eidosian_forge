from fontTools import ttLib
from fontTools.misc.textTools import safeEval
from fontTools.ttLib.tables.DefaultTable import DefaultTable
import sys
import os
import logging
def _parseFile(self, file):
    from xml.parsers.expat import ParserCreate
    parser = ParserCreate()
    parser.StartElementHandler = self._startElementHandler
    parser.EndElementHandler = self._endElementHandler
    parser.CharacterDataHandler = self._characterDataHandler
    pos = 0
    while True:
        chunk = file.read(BUFSIZE)
        if not chunk:
            parser.Parse(chunk, 1)
            break
        pos = pos + len(chunk)
        if self.progress:
            self.progress.set(pos // 100)
        parser.Parse(chunk, 0)