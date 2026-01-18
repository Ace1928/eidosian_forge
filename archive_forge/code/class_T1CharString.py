from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, strjoin
from fontTools.pens.boundsPen import BoundsPen
import struct
import logging
class T1CharString(T2CharString):
    operandEncoding = t1OperandEncoding
    operators, opcodes = buildOperatorDict(t1Operators)

    def __init__(self, bytecode=None, program=None, subrs=None):
        super().__init__(bytecode, program)
        self.subrs = subrs

    def getIntEncoder(self):
        return encodeIntT1

    def getFixedEncoder(self):

        def encodeFixed(value):
            raise TypeError("Type 1 charstrings don't support floating point operands")

    def decompile(self):
        if self.bytecode is None:
            return
        program = []
        index = 0
        while True:
            token, isOperator, index = self.getToken(index)
            if token is None:
                break
            program.append(token)
        self.setProgram(program)

    def draw(self, pen):
        extractor = T1OutlineExtractor(pen, self.subrs)
        extractor.execute(self)
        self.width = extractor.width