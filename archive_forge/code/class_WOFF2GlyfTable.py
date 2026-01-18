from io import BytesIO
import sys
import array
import struct
from collections import OrderedDict
from fontTools.misc import sstruct
from fontTools.misc.arrayTools import calcIntBounds
from fontTools.misc.textTools import Tag, bytechr, byteord, bytesjoin, pad
from fontTools.ttLib import (
from fontTools.ttLib.sfnt import (
from fontTools.ttLib.tables import ttProgram, _g_l_y_f
import logging
class WOFF2GlyfTable(getTableClass('glyf')):
    """Decoder/Encoder for WOFF2 'glyf' table transform."""
    subStreams = ('nContourStream', 'nPointsStream', 'flagStream', 'glyphStream', 'compositeStream', 'bboxStream', 'instructionStream')

    def __init__(self, tag=None):
        self.tableTag = Tag(tag or 'glyf')

    def reconstruct(self, data, ttFont):
        """Decompile transformed 'glyf' data."""
        inputDataSize = len(data)
        if inputDataSize < woff2GlyfTableFormatSize:
            raise TTLibError("not enough 'glyf' data")
        dummy, data = sstruct.unpack2(woff2GlyfTableFormat, data, self)
        offset = woff2GlyfTableFormatSize
        for stream in self.subStreams:
            size = getattr(self, stream + 'Size')
            setattr(self, stream, data[:size])
            data = data[size:]
            offset += size
        hasOverlapSimpleBitmap = self.optionFlags & woff2OverlapSimpleBitmapFlag
        self.overlapSimpleBitmap = None
        if hasOverlapSimpleBitmap:
            overlapSimpleBitmapSize = self.numGlyphs + 7 >> 3
            self.overlapSimpleBitmap = array.array('B', data[:overlapSimpleBitmapSize])
            offset += overlapSimpleBitmapSize
        if offset != inputDataSize:
            raise TTLibError("incorrect size of transformed 'glyf' table: expected %d, received %d bytes" % (offset, inputDataSize))
        bboxBitmapSize = self.numGlyphs + 31 >> 5 << 2
        bboxBitmap = self.bboxStream[:bboxBitmapSize]
        self.bboxBitmap = array.array('B', bboxBitmap)
        self.bboxStream = self.bboxStream[bboxBitmapSize:]
        self.nContourStream = array.array('h', self.nContourStream)
        if sys.byteorder != 'big':
            self.nContourStream.byteswap()
        assert len(self.nContourStream) == self.numGlyphs
        if 'head' in ttFont:
            ttFont['head'].indexToLocFormat = self.indexFormat
        try:
            self.glyphOrder = ttFont.getGlyphOrder()
        except:
            self.glyphOrder = None
        if self.glyphOrder is None:
            self.glyphOrder = ['.notdef']
            self.glyphOrder.extend(['glyph%.5d' % i for i in range(1, self.numGlyphs)])
        elif len(self.glyphOrder) != self.numGlyphs:
            raise TTLibError('incorrect glyphOrder: expected %d glyphs, found %d' % (len(self.glyphOrder), self.numGlyphs))
        glyphs = self.glyphs = {}
        for glyphID, glyphName in enumerate(self.glyphOrder):
            glyph = self._decodeGlyph(glyphID)
            glyphs[glyphName] = glyph

    def transform(self, ttFont):
        """Return transformed 'glyf' data"""
        self.numGlyphs = len(self.glyphs)
        assert len(self.glyphOrder) == self.numGlyphs
        if 'maxp' in ttFont:
            ttFont['maxp'].numGlyphs = self.numGlyphs
        self.indexFormat = ttFont['head'].indexToLocFormat
        for stream in self.subStreams:
            setattr(self, stream, b'')
        bboxBitmapSize = self.numGlyphs + 31 >> 5 << 2
        self.bboxBitmap = array.array('B', [0] * bboxBitmapSize)
        self.overlapSimpleBitmap = array.array('B', [0] * (self.numGlyphs + 7 >> 3))
        for glyphID in range(self.numGlyphs):
            try:
                self._encodeGlyph(glyphID)
            except NotImplementedError:
                return None
        hasOverlapSimpleBitmap = any(self.overlapSimpleBitmap)
        self.bboxStream = self.bboxBitmap.tobytes() + self.bboxStream
        for stream in self.subStreams:
            setattr(self, stream + 'Size', len(getattr(self, stream)))
        self.version = 0
        self.optionFlags = 0
        if hasOverlapSimpleBitmap:
            self.optionFlags |= woff2OverlapSimpleBitmapFlag
        data = sstruct.pack(woff2GlyfTableFormat, self)
        data += bytesjoin([getattr(self, s) for s in self.subStreams])
        if hasOverlapSimpleBitmap:
            data += self.overlapSimpleBitmap.tobytes()
        return data

    def _decodeGlyph(self, glyphID):
        glyph = getTableModule('glyf').Glyph()
        glyph.numberOfContours = self.nContourStream[glyphID]
        if glyph.numberOfContours == 0:
            return glyph
        elif glyph.isComposite():
            self._decodeComponents(glyph)
        else:
            self._decodeCoordinates(glyph)
            self._decodeOverlapSimpleFlag(glyph, glyphID)
        self._decodeBBox(glyphID, glyph)
        return glyph

    def _decodeComponents(self, glyph):
        data = self.compositeStream
        glyph.components = []
        more = 1
        haveInstructions = 0
        while more:
            component = getTableModule('glyf').GlyphComponent()
            more, haveInstr, data = component.decompile(data, self)
            haveInstructions = haveInstructions | haveInstr
            glyph.components.append(component)
        self.compositeStream = data
        if haveInstructions:
            self._decodeInstructions(glyph)

    def _decodeCoordinates(self, glyph):
        data = self.nPointsStream
        endPtsOfContours = []
        endPoint = -1
        for i in range(glyph.numberOfContours):
            ptsOfContour, data = unpack255UShort(data)
            endPoint += ptsOfContour
            endPtsOfContours.append(endPoint)
        glyph.endPtsOfContours = endPtsOfContours
        self.nPointsStream = data
        self._decodeTriplets(glyph)
        self._decodeInstructions(glyph)

    def _decodeOverlapSimpleFlag(self, glyph, glyphID):
        if self.overlapSimpleBitmap is None or glyph.numberOfContours <= 0:
            return
        byte = glyphID >> 3
        bit = glyphID & 7
        if self.overlapSimpleBitmap[byte] & 128 >> bit:
            glyph.flags[0] |= _g_l_y_f.flagOverlapSimple

    def _decodeInstructions(self, glyph):
        glyphStream = self.glyphStream
        instructionStream = self.instructionStream
        instructionLength, glyphStream = unpack255UShort(glyphStream)
        glyph.program = ttProgram.Program()
        glyph.program.fromBytecode(instructionStream[:instructionLength])
        self.glyphStream = glyphStream
        self.instructionStream = instructionStream[instructionLength:]

    def _decodeBBox(self, glyphID, glyph):
        haveBBox = bool(self.bboxBitmap[glyphID >> 3] & 128 >> (glyphID & 7))
        if glyph.isComposite() and (not haveBBox):
            raise TTLibError('no bbox values for composite glyph %d' % glyphID)
        if haveBBox:
            dummy, self.bboxStream = sstruct.unpack2(bboxFormat, self.bboxStream, glyph)
        else:
            glyph.recalcBounds(self)

    def _decodeTriplets(self, glyph):

        def withSign(flag, baseval):
            assert 0 <= baseval and baseval < 65536, 'integer overflow'
            return baseval if flag & 1 else -baseval
        nPoints = glyph.endPtsOfContours[-1] + 1
        flagSize = nPoints
        if flagSize > len(self.flagStream):
            raise TTLibError("not enough 'flagStream' data")
        flagsData = self.flagStream[:flagSize]
        self.flagStream = self.flagStream[flagSize:]
        flags = array.array('B', flagsData)
        triplets = array.array('B', self.glyphStream)
        nTriplets = len(triplets)
        assert nPoints <= nTriplets
        x = 0
        y = 0
        glyph.coordinates = getTableModule('glyf').GlyphCoordinates.zeros(nPoints)
        glyph.flags = array.array('B')
        tripletIndex = 0
        for i in range(nPoints):
            flag = flags[i]
            onCurve = not bool(flag >> 7)
            flag &= 127
            if flag < 84:
                nBytes = 1
            elif flag < 120:
                nBytes = 2
            elif flag < 124:
                nBytes = 3
            else:
                nBytes = 4
            assert tripletIndex + nBytes <= nTriplets
            if flag < 10:
                dx = 0
                dy = withSign(flag, ((flag & 14) << 7) + triplets[tripletIndex])
            elif flag < 20:
                dx = withSign(flag, ((flag - 10 & 14) << 7) + triplets[tripletIndex])
                dy = 0
            elif flag < 84:
                b0 = flag - 20
                b1 = triplets[tripletIndex]
                dx = withSign(flag, 1 + (b0 & 48) + (b1 >> 4))
                dy = withSign(flag >> 1, 1 + ((b0 & 12) << 2) + (b1 & 15))
            elif flag < 120:
                b0 = flag - 84
                dx = withSign(flag, 1 + (b0 // 12 << 8) + triplets[tripletIndex])
                dy = withSign(flag >> 1, 1 + (b0 % 12 >> 2 << 8) + triplets[tripletIndex + 1])
            elif flag < 124:
                b2 = triplets[tripletIndex + 1]
                dx = withSign(flag, (triplets[tripletIndex] << 4) + (b2 >> 4))
                dy = withSign(flag >> 1, ((b2 & 15) << 8) + triplets[tripletIndex + 2])
            else:
                dx = withSign(flag, (triplets[tripletIndex] << 8) + triplets[tripletIndex + 1])
                dy = withSign(flag >> 1, (triplets[tripletIndex + 2] << 8) + triplets[tripletIndex + 3])
            tripletIndex += nBytes
            x += dx
            y += dy
            glyph.coordinates[i] = (x, y)
            glyph.flags.append(int(onCurve))
        bytesConsumed = tripletIndex
        self.glyphStream = self.glyphStream[bytesConsumed:]

    def _encodeGlyph(self, glyphID):
        glyphName = self.getGlyphName(glyphID)
        glyph = self[glyphName]
        self.nContourStream += struct.pack('>h', glyph.numberOfContours)
        if glyph.numberOfContours == 0:
            return
        elif glyph.isComposite():
            self._encodeComponents(glyph)
        elif glyph.isVarComposite():
            raise NotImplementedError
        else:
            self._encodeCoordinates(glyph)
            self._encodeOverlapSimpleFlag(glyph, glyphID)
        self._encodeBBox(glyphID, glyph)

    def _encodeComponents(self, glyph):
        lastcomponent = len(glyph.components) - 1
        more = 1
        haveInstructions = 0
        for i in range(len(glyph.components)):
            if i == lastcomponent:
                haveInstructions = hasattr(glyph, 'program')
                more = 0
            component = glyph.components[i]
            self.compositeStream += component.compile(more, haveInstructions, self)
        if haveInstructions:
            self._encodeInstructions(glyph)

    def _encodeCoordinates(self, glyph):
        lastEndPoint = -1
        if _g_l_y_f.flagCubic in glyph.flags:
            raise NotImplementedError
        for endPoint in glyph.endPtsOfContours:
            ptsOfContour = endPoint - lastEndPoint
            self.nPointsStream += pack255UShort(ptsOfContour)
            lastEndPoint = endPoint
        self._encodeTriplets(glyph)
        self._encodeInstructions(glyph)

    def _encodeOverlapSimpleFlag(self, glyph, glyphID):
        if glyph.numberOfContours <= 0:
            return
        if glyph.flags[0] & _g_l_y_f.flagOverlapSimple:
            byte = glyphID >> 3
            bit = glyphID & 7
            self.overlapSimpleBitmap[byte] |= 128 >> bit

    def _encodeInstructions(self, glyph):
        instructions = glyph.program.getBytecode()
        self.glyphStream += pack255UShort(len(instructions))
        self.instructionStream += instructions

    def _encodeBBox(self, glyphID, glyph):
        assert glyph.numberOfContours != 0, 'empty glyph has no bbox'
        if not glyph.isComposite():
            currentBBox = (glyph.xMin, glyph.yMin, glyph.xMax, glyph.yMax)
            calculatedBBox = calcIntBounds(glyph.coordinates)
            if currentBBox == calculatedBBox:
                return
        self.bboxBitmap[glyphID >> 3] |= 128 >> (glyphID & 7)
        self.bboxStream += sstruct.pack(bboxFormat, glyph)

    def _encodeTriplets(self, glyph):
        assert len(glyph.coordinates) == len(glyph.flags)
        coordinates = glyph.coordinates.copy()
        coordinates.absoluteToRelative()
        flags = array.array('B')
        triplets = array.array('B')
        for i in range(len(coordinates)):
            onCurve = glyph.flags[i] & _g_l_y_f.flagOnCurve
            x, y = coordinates[i]
            absX = abs(x)
            absY = abs(y)
            onCurveBit = 0 if onCurve else 128
            xSignBit = 0 if x < 0 else 1
            ySignBit = 0 if y < 0 else 1
            xySignBits = xSignBit + 2 * ySignBit
            if x == 0 and absY < 1280:
                flags.append(onCurveBit + ((absY & 3840) >> 7) + ySignBit)
                triplets.append(absY & 255)
            elif y == 0 and absX < 1280:
                flags.append(onCurveBit + 10 + ((absX & 3840) >> 7) + xSignBit)
                triplets.append(absX & 255)
            elif absX < 65 and absY < 65:
                flags.append(onCurveBit + 20 + (absX - 1 & 48) + ((absY - 1 & 48) >> 2) + xySignBits)
                triplets.append((absX - 1 & 15) << 4 | absY - 1 & 15)
            elif absX < 769 and absY < 769:
                flags.append(onCurveBit + 84 + 12 * ((absX - 1 & 768) >> 8) + ((absY - 1 & 768) >> 6) + xySignBits)
                triplets.append(absX - 1 & 255)
                triplets.append(absY - 1 & 255)
            elif absX < 4096 and absY < 4096:
                flags.append(onCurveBit + 120 + xySignBits)
                triplets.append(absX >> 4)
                triplets.append((absX & 15) << 4 | absY >> 8)
                triplets.append(absY & 255)
            else:
                flags.append(onCurveBit + 124 + xySignBits)
                triplets.append(absX >> 8)
                triplets.append(absX & 255)
                triplets.append(absY >> 8)
                triplets.append(absY & 255)
        self.flagStream += flags.tobytes()
        self.glyphStream += triplets.tobytes()