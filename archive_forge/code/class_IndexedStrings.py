from fontTools.misc import sstruct
from fontTools.misc import psCharStrings
from fontTools.misc.arrayTools import unionRect, intRect
from fontTools.misc.textTools import (
from fontTools.ttLib import TTFont
from fontTools.ttLib.tables.otBase import OTTableWriter
from fontTools.ttLib.tables.otBase import OTTableReader
from fontTools.ttLib.tables import otTables as ot
from io import BytesIO
import struct
import logging
import re
class IndexedStrings(object):
    """SID -> string mapping."""

    def __init__(self, file=None):
        if file is None:
            strings = []
        else:
            strings = [tostr(s, encoding='latin1') for s in Index(file, isCFF2=False)]
        self.strings = strings

    def getCompiler(self):
        return IndexedStringsCompiler(self, None, self, isCFF2=False)

    def __len__(self):
        return len(self.strings)

    def __getitem__(self, SID):
        if SID < cffStandardStringCount:
            return cffStandardStrings[SID]
        else:
            return self.strings[SID - cffStandardStringCount]

    def getSID(self, s):
        if not hasattr(self, 'stringMapping'):
            self.buildStringMapping()
        s = tostr(s, encoding='latin1')
        if s in cffStandardStringMapping:
            SID = cffStandardStringMapping[s]
        elif s in self.stringMapping:
            SID = self.stringMapping[s]
        else:
            SID = len(self.strings) + cffStandardStringCount
            self.strings.append(s)
            self.stringMapping[s] = SID
        return SID

    def getStrings(self):
        return self.strings

    def buildStringMapping(self):
        self.stringMapping = {}
        for index in range(len(self.strings)):
            self.stringMapping[self.strings[index]] = index + cffStandardStringCount