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
class VarStoreData(object):

    def __init__(self, file=None, otVarStore=None):
        self.file = file
        self.data = None
        self.otVarStore = otVarStore
        self.font = TTFont()

    def decompile(self):
        if self.file:
            length = readCard16(self.file)
            self.data = self.file.read(length)
            globalState = {}
            reader = OTTableReader(self.data, globalState)
            self.otVarStore = ot.VarStore()
            self.otVarStore.decompile(reader, self.font)
        return self

    def compile(self):
        writer = OTTableWriter()
        self.otVarStore.compile(writer, self.font)
        self.data = writer.getAllData()

    def writeXML(self, xmlWriter, name):
        self.otVarStore.toXML(xmlWriter, self.font)

    def xmlRead(self, name, attrs, content, parent):
        self.otVarStore = ot.VarStore()
        for element in content:
            if isinstance(element, tuple):
                name, attrs, content = element
                self.otVarStore.fromXML(name, attrs, content, self.font)
            else:
                pass
        return None

    def __len__(self):
        return len(self.data)

    def getNumRegions(self, vsIndex):
        if vsIndex is None:
            vsIndex = 0
        varData = self.otVarStore.VarData[vsIndex]
        numRegions = varData.VarRegionCount
        return numRegions