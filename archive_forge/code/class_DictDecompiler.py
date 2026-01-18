from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, strjoin
from fontTools.pens.boundsPen import BoundsPen
import struct
import logging
class DictDecompiler(object):
    operandEncoding = cffDictOperandEncoding

    def __init__(self, strings, parent=None):
        self.stack = []
        self.strings = strings
        self.dict = {}
        self.parent = parent

    def getDict(self):
        assert len(self.stack) == 0, 'non-empty stack'
        return self.dict

    def decompile(self, data):
        index = 0
        lenData = len(data)
        push = self.stack.append
        while index < lenData:
            b0 = byteord(data[index])
            index = index + 1
            handler = self.operandEncoding[b0]
            value, index = handler(self, b0, data, index)
            if value is not None:
                push(value)

    def pop(self):
        value = self.stack[-1]
        del self.stack[-1]
        return value

    def popall(self):
        args = self.stack[:]
        del self.stack[:]
        return args

    def handle_operator(self, operator):
        operator, argType = operator
        if isinstance(argType, tuple):
            value = ()
            for i in range(len(argType) - 1, -1, -1):
                arg = argType[i]
                arghandler = getattr(self, 'arg_' + arg)
                value = (arghandler(operator),) + value
        else:
            arghandler = getattr(self, 'arg_' + argType)
            value = arghandler(operator)
        if operator == 'blend':
            self.stack.extend(value)
        else:
            self.dict[operator] = value

    def arg_number(self, name):
        if isinstance(self.stack[0], list):
            out = self.arg_blend_number(self.stack)
        else:
            out = self.pop()
        return out

    def arg_blend_number(self, name):
        out = []
        blendArgs = self.pop()
        numMasters = len(blendArgs)
        out.append(blendArgs)
        out.append('blend')
        dummy = self.popall()
        return blendArgs

    def arg_SID(self, name):
        return self.strings[self.pop()]

    def arg_array(self, name):
        return self.popall()

    def arg_blendList(self, name):
        """
        There may be non-blend args at the top of the stack. We first calculate
        where the blend args start in the stack. These are the last
        numMasters*numBlends) +1 args.
        The blend args starts with numMasters relative coordinate values, the  BlueValues in the list from the default master font. This is followed by
        numBlends list of values. Each of  value in one of these lists is the
        Variable Font delta for the matching region.

        We re-arrange this to be a list of numMaster entries. Each entry starts with the corresponding default font relative value, and is followed by
        the delta values. We then convert the default values, the first item in each entry, to an absolute value.
        """
        vsindex = self.dict.get('vsindex', 0)
        numMasters = self.parent.getNumRegions(vsindex) + 1
        numBlends = self.pop()
        args = self.popall()
        numArgs = len(args)
        assert numArgs == numMasters * numBlends
        value = [None] * numBlends
        numDeltas = numMasters - 1
        i = 0
        prevVal = 0
        while i < numBlends:
            newVal = args[i] + prevVal
            prevVal = newVal
            masterOffset = numBlends + i * numDeltas
            blendList = [newVal] + args[masterOffset:masterOffset + numDeltas]
            value[i] = blendList
            i += 1
        return value

    def arg_delta(self, name):
        valueList = self.popall()
        out = []
        if valueList and isinstance(valueList[0], list):
            out = valueList
        else:
            current = 0
            for v in valueList:
                current = current + v
                out.append(current)
        return out