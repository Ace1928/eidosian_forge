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
class CharStringsConverter(TableConverter):

    def _read(self, parent, value):
        file = parent.file
        isCFF2 = parent._isCFF2
        charset = parent.charset
        varStore = getattr(parent, 'VarStore', None)
        globalSubrs = parent.GlobalSubrs
        if hasattr(parent, 'FDArray'):
            fdArray = parent.FDArray
            if hasattr(parent, 'FDSelect'):
                fdSelect = parent.FDSelect
            else:
                fdSelect = None
            private = None
        else:
            fdSelect, fdArray = (None, None)
            private = parent.Private
        file.seek(value)
        charStrings = CharStrings(file, charset, globalSubrs, private, fdSelect, fdArray, isCFF2=isCFF2, varStore=varStore)
        return charStrings

    def write(self, parent, value):
        return 0

    def xmlRead(self, name, attrs, content, parent):
        if hasattr(parent, 'FDArray'):
            fdArray = parent.FDArray
            if hasattr(parent, 'FDSelect'):
                fdSelect = parent.FDSelect
            else:
                fdSelect = None
            private = None
        else:
            private, fdSelect, fdArray = (parent.Private, None, None)
        charStrings = CharStrings(None, None, parent.GlobalSubrs, private, fdSelect, fdArray, varStore=getattr(parent, 'VarStore', None))
        charStrings.fromXML(name, attrs, content)
        return charStrings