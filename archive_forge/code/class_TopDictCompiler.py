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
class TopDictCompiler(DictCompiler):
    opcodes = buildOpcodeDict(topDictOperators)

    def getChildren(self, strings):
        isCFF2 = self.isCFF2
        children = []
        if self.dictObj.cff2GetGlyphOrder is None:
            if hasattr(self.dictObj, 'charset') and self.dictObj.charset:
                if hasattr(self.dictObj, 'ROS'):
                    charsetCode = None
                else:
                    charsetCode = getStdCharSet(self.dictObj.charset)
                if charsetCode is None:
                    children.append(CharsetCompiler(strings, self.dictObj.charset, self))
                else:
                    self.rawDict['charset'] = charsetCode
            if hasattr(self.dictObj, 'Encoding') and self.dictObj.Encoding:
                encoding = self.dictObj.Encoding
                if not isinstance(encoding, str):
                    children.append(EncodingCompiler(strings, encoding, self))
        elif hasattr(self.dictObj, 'VarStore'):
            varStoreData = self.dictObj.VarStore
            varStoreComp = VarStoreCompiler(varStoreData, self)
            children.append(varStoreComp)
        if hasattr(self.dictObj, 'FDSelect'):
            fdSelect = self.dictObj.FDSelect
            if len(fdSelect) == 0:
                charStrings = self.dictObj.CharStrings
                for name in self.dictObj.charset:
                    fdSelect.append(charStrings[name].fdSelectIndex)
            fdSelectComp = FDSelectCompiler(fdSelect, self)
            children.append(fdSelectComp)
        if hasattr(self.dictObj, 'CharStrings'):
            items = []
            charStrings = self.dictObj.CharStrings
            for name in self.dictObj.charset:
                items.append(charStrings[name])
            charStringsComp = CharStringsCompiler(items, strings, self, isCFF2=isCFF2)
            children.append(charStringsComp)
        if hasattr(self.dictObj, 'FDArray'):
            fdArrayIndexComp = self.dictObj.FDArray.getCompiler(strings, self)
            children.append(fdArrayIndexComp)
            children.extend(fdArrayIndexComp.getChildren(strings))
        if hasattr(self.dictObj, 'Private'):
            privComp = self.dictObj.Private.getCompiler(strings, self)
            children.append(privComp)
            children.extend(privComp.getChildren(strings))
        return children