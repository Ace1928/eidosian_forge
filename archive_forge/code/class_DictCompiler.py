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
class DictCompiler(object):
    maxBlendStack = 0

    def __init__(self, dictObj, strings, parent, isCFF2=None):
        if strings:
            assert isinstance(strings, IndexedStrings)
        if isCFF2 is None and hasattr(parent, 'isCFF2'):
            isCFF2 = parent.isCFF2
            assert isCFF2 is not None
        self.isCFF2 = isCFF2
        self.dictObj = dictObj
        self.strings = strings
        self.parent = parent
        rawDict = {}
        for name in dictObj.order:
            value = getattr(dictObj, name, None)
            if value is None:
                continue
            conv = dictObj.converters[name]
            value = conv.write(dictObj, value)
            if value == dictObj.defaults.get(name):
                continue
            rawDict[name] = value
        self.rawDict = rawDict

    def setPos(self, pos, endPos):
        pass

    def getDataLength(self):
        return len(self.compile('getDataLength'))

    def compile(self, reason):
        log.log(DEBUG, '-- compiling %s for %s', self.__class__.__name__, reason)
        rawDict = self.rawDict
        data = []
        for name in self.dictObj.order:
            value = rawDict.get(name)
            if value is None:
                continue
            op, argType = self.opcodes[name]
            if isinstance(argType, tuple):
                l = len(argType)
                assert len(value) == l, "value doesn't match arg type"
                for i in range(l):
                    arg = argType[i]
                    v = value[i]
                    arghandler = getattr(self, 'arg_' + arg)
                    data.append(arghandler(v))
            else:
                arghandler = getattr(self, 'arg_' + argType)
                data.append(arghandler(value))
            data.append(op)
        data = bytesjoin(data)
        return data

    def toFile(self, file):
        data = self.compile('toFile')
        file.write(data)

    def arg_number(self, num):
        if isinstance(num, list):
            data = [encodeNumber(val) for val in num]
            data.append(encodeNumber(1))
            data.append(bytechr(blendOp))
            datum = bytesjoin(data)
        else:
            datum = encodeNumber(num)
        return datum

    def arg_SID(self, s):
        return psCharStrings.encodeIntCFF(self.strings.getSID(s))

    def arg_array(self, value):
        data = []
        for num in value:
            data.append(self.arg_number(num))
        return bytesjoin(data)

    def arg_delta(self, value):
        if not value:
            return b''
        val0 = value[0]
        if isinstance(val0, list):
            data = self.arg_delta_blend(value)
        else:
            out = []
            last = 0
            for v in value:
                out.append(v - last)
                last = v
            data = []
            for num in out:
                data.append(encodeNumber(num))
        return bytesjoin(data)

    def arg_delta_blend(self, value):
        """A delta list with blend lists has to be *all* blend lists.

        The value is a list is arranged as follows::

                [
                        [V0, d0..dn]
                        [V1, d0..dn]
                        ...
                        [Vm, d0..dn]
                ]

        ``V`` is the absolute coordinate value from the default font, and ``d0-dn``
        are the delta values from the *n* regions. Each ``V`` is an absolute
        coordinate from the default font.

        We want to return a list::

                [
                        [v0, v1..vm]
                        [d0..dn]
                        ...
                        [d0..dn]
                        numBlends
                        blendOp
                ]

        where each ``v`` is relative to the previous default font value.
        """
        numMasters = len(value[0])
        numBlends = len(value)
        numStack = numBlends * numMasters + 1
        if numStack > self.maxBlendStack:
            numBlendValues = int((self.maxBlendStack - 1) / numMasters)
            out = []
            while True:
                numVal = min(len(value), numBlendValues)
                if numVal == 0:
                    break
                valList = value[0:numVal]
                out1 = self.arg_delta_blend(valList)
                out.extend(out1)
                value = value[numVal:]
        else:
            firstList = [0] * numBlends
            deltaList = [None] * numBlends
            i = 0
            prevVal = 0
            while i < numBlends:
                defaultValue = value[i][0]
                firstList[i] = defaultValue - prevVal
                prevVal = defaultValue
                deltaList[i] = value[i][1:]
                i += 1
            relValueList = firstList
            for blendList in deltaList:
                relValueList.extend(blendList)
            out = [encodeNumber(val) for val in relValueList]
            out.append(encodeNumber(numBlends))
            out.append(bytechr(blendOp))
        return out