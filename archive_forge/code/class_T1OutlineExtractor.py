from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, strjoin
from fontTools.pens.boundsPen import BoundsPen
import struct
import logging
class T1OutlineExtractor(T2OutlineExtractor):

    def __init__(self, pen, subrs):
        self.pen = pen
        self.subrs = subrs
        self.reset()

    def reset(self):
        self.flexing = 0
        self.width = 0
        self.sbx = 0
        T2OutlineExtractor.reset(self)

    def endPath(self):
        if self.sawMoveTo:
            self.pen.endPath()
        self.sawMoveTo = 0

    def popallWidth(self, evenOdd=0):
        return self.popall()

    def exch(self):
        stack = self.operandStack
        stack[-1], stack[-2] = (stack[-2], stack[-1])

    def op_rmoveto(self, index):
        if self.flexing:
            return
        self.endPath()
        self.rMoveTo(self.popall())

    def op_hmoveto(self, index):
        if self.flexing:
            self.push(0)
            return
        self.endPath()
        self.rMoveTo((self.popall()[0], 0))

    def op_vmoveto(self, index):
        if self.flexing:
            self.push(0)
            self.exch()
            return
        self.endPath()
        self.rMoveTo((0, self.popall()[0]))

    def op_closepath(self, index):
        self.closePath()

    def op_setcurrentpoint(self, index):
        args = self.popall()
        x, y = args
        self.currentPoint = (x, y)

    def op_endchar(self, index):
        self.endPath()

    def op_hsbw(self, index):
        sbx, wx = self.popall()
        self.width = wx
        self.sbx = sbx
        self.currentPoint = (sbx, self.currentPoint[1])

    def op_sbw(self, index):
        self.popall()

    def op_callsubr(self, index):
        subrIndex = self.pop()
        subr = self.subrs[subrIndex]
        self.execute(subr)

    def op_callothersubr(self, index):
        subrIndex = self.pop()
        nArgs = self.pop()
        if subrIndex == 0 and nArgs == 3:
            self.doFlex()
            self.flexing = 0
        elif subrIndex == 1 and nArgs == 0:
            self.flexing = 1

    def op_pop(self, index):
        pass

    def doFlex(self):
        finaly = self.pop()
        finalx = self.pop()
        self.pop()
        p3y = self.pop()
        p3x = self.pop()
        bcp4y = self.pop()
        bcp4x = self.pop()
        bcp3y = self.pop()
        bcp3x = self.pop()
        p2y = self.pop()
        p2x = self.pop()
        bcp2y = self.pop()
        bcp2x = self.pop()
        bcp1y = self.pop()
        bcp1x = self.pop()
        rpy = self.pop()
        rpx = self.pop()
        self.push(bcp1x + rpx)
        self.push(bcp1y + rpy)
        self.push(bcp2x)
        self.push(bcp2y)
        self.push(p2x)
        self.push(p2y)
        self.op_rrcurveto(None)
        self.push(bcp3x)
        self.push(bcp3y)
        self.push(bcp4x)
        self.push(bcp4y)
        self.push(p3x)
        self.push(p3y)
        self.op_rrcurveto(None)
        self.push(finalx)
        self.push(finaly)

    def op_dotsection(self, index):
        self.popall()

    def op_hstem3(self, index):
        self.popall()

    def op_seac(self, index):
        """asb adx ady bchar achar seac"""
        from fontTools.encodings.StandardEncoding import StandardEncoding
        asb, adx, ady, bchar, achar = self.popall()
        baseGlyph = StandardEncoding[bchar]
        self.pen.addComponent(baseGlyph, (1, 0, 0, 1, 0, 0))
        accentGlyph = StandardEncoding[achar]
        adx = adx + self.sbx - asb
        self.pen.addComponent(accentGlyph, (1, 0, 0, 1, adx, ady))

    def op_vstem3(self, index):
        self.popall()