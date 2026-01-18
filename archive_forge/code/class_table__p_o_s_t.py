from fontTools import ttLib
from fontTools.ttLib.standardGlyphOrder import standardGlyphOrder
from fontTools.misc import sstruct
from fontTools.misc.textTools import bytechr, byteord, tobytes, tostr, safeEval, readHex
from . import DefaultTable
import sys
import struct
import array
import logging
class table__p_o_s_t(DefaultTable.DefaultTable):

    def decompile(self, data, ttFont):
        sstruct.unpack(postFormat, data[:postFormatSize], self)
        data = data[postFormatSize:]
        if self.formatType == 1.0:
            self.decode_format_1_0(data, ttFont)
        elif self.formatType == 2.0:
            self.decode_format_2_0(data, ttFont)
        elif self.formatType == 3.0:
            self.decode_format_3_0(data, ttFont)
        elif self.formatType == 4.0:
            self.decode_format_4_0(data, ttFont)
        else:
            raise ttLib.TTLibError("'post' table format %f not supported" % self.formatType)

    def compile(self, ttFont):
        data = sstruct.pack(postFormat, self)
        if self.formatType == 1.0:
            pass
        elif self.formatType == 2.0:
            data = data + self.encode_format_2_0(ttFont)
        elif self.formatType == 3.0:
            pass
        elif self.formatType == 4.0:
            data = data + self.encode_format_4_0(ttFont)
        else:
            raise ttLib.TTLibError("'post' table format %f not supported" % self.formatType)
        return data

    def getGlyphOrder(self):
        """This function will get called by a ttLib.TTFont instance.
        Do not call this function yourself, use TTFont().getGlyphOrder()
        or its relatives instead!
        """
        if not hasattr(self, 'glyphOrder'):
            raise ttLib.TTLibError('illegal use of getGlyphOrder()')
        glyphOrder = self.glyphOrder
        del self.glyphOrder
        return glyphOrder

    def decode_format_1_0(self, data, ttFont):
        self.glyphOrder = standardGlyphOrder[:ttFont['maxp'].numGlyphs]

    def decode_format_2_0(self, data, ttFont):
        numGlyphs, = struct.unpack('>H', data[:2])
        numGlyphs = int(numGlyphs)
        if numGlyphs > ttFont['maxp'].numGlyphs:
            numGlyphs = ttFont['maxp'].numGlyphs
        data = data[2:]
        indices = array.array('H')
        indices.frombytes(data[:2 * numGlyphs])
        if sys.byteorder != 'big':
            indices.byteswap()
        data = data[2 * numGlyphs:]
        maxIndex = max(indices)
        self.extraNames = extraNames = unpackPStrings(data, maxIndex - 257)
        self.glyphOrder = glyphOrder = [''] * int(ttFont['maxp'].numGlyphs)
        for glyphID in range(numGlyphs):
            index = indices[glyphID]
            if index > 257:
                try:
                    name = extraNames[index - 258]
                except IndexError:
                    name = ''
            else:
                name = standardGlyphOrder[index]
            glyphOrder[glyphID] = name
        self.build_psNameMapping(ttFont)

    def build_psNameMapping(self, ttFont):
        mapping = {}
        allNames = {}
        for i in range(ttFont['maxp'].numGlyphs):
            glyphName = psName = self.glyphOrder[i]
            if glyphName == '':
                glyphName = 'glyph%.5d' % i
            if glyphName in allNames:
                n = allNames[glyphName]
                while glyphName + '#' + str(n) in allNames:
                    n += 1
                allNames[glyphName] = n + 1
                glyphName = glyphName + '#' + str(n)
            self.glyphOrder[i] = glyphName
            allNames[glyphName] = 1
            if glyphName != psName:
                mapping[glyphName] = psName
        self.mapping = mapping

    def decode_format_3_0(self, data, ttFont):
        self.glyphOrder = None

    def decode_format_4_0(self, data, ttFont):
        from fontTools import agl
        numGlyphs = ttFont['maxp'].numGlyphs
        indices = array.array('H')
        indices.frombytes(data)
        if sys.byteorder != 'big':
            indices.byteswap()
        self.glyphOrder = glyphOrder = [''] * int(numGlyphs)
        for i in range(min(len(indices), numGlyphs)):
            if indices[i] == 65535:
                self.glyphOrder[i] = ''
            elif indices[i] in agl.UV2AGL:
                self.glyphOrder[i] = agl.UV2AGL[indices[i]]
            else:
                self.glyphOrder[i] = 'uni%04X' % indices[i]
        self.build_psNameMapping(ttFont)

    def encode_format_2_0(self, ttFont):
        numGlyphs = ttFont['maxp'].numGlyphs
        glyphOrder = ttFont.getGlyphOrder()
        assert len(glyphOrder) == numGlyphs
        indices = array.array('H')
        extraDict = {}
        extraNames = self.extraNames = [n for n in self.extraNames if n not in standardGlyphOrder]
        for i in range(len(extraNames)):
            extraDict[extraNames[i]] = i
        for glyphID in range(numGlyphs):
            glyphName = glyphOrder[glyphID]
            if glyphName in self.mapping:
                psName = self.mapping[glyphName]
            else:
                psName = glyphName
            if psName in extraDict:
                index = 258 + extraDict[psName]
            elif psName in standardGlyphOrder:
                index = standardGlyphOrder.index(psName)
            else:
                index = 258 + len(extraNames)
                extraDict[psName] = len(extraNames)
                extraNames.append(psName)
            indices.append(index)
        if sys.byteorder != 'big':
            indices.byteswap()
        return struct.pack('>H', numGlyphs) + indices.tobytes() + packPStrings(extraNames)

    def encode_format_4_0(self, ttFont):
        from fontTools import agl
        numGlyphs = ttFont['maxp'].numGlyphs
        glyphOrder = ttFont.getGlyphOrder()
        assert len(glyphOrder) == numGlyphs
        indices = array.array('H')
        for glyphID in glyphOrder:
            glyphID = glyphID.split('#')[0]
            if glyphID in agl.AGL2UV:
                indices.append(agl.AGL2UV[glyphID])
            elif len(glyphID) == 7 and glyphID[:3] == 'uni':
                indices.append(int(glyphID[3:], 16))
            else:
                indices.append(65535)
        if sys.byteorder != 'big':
            indices.byteswap()
        return indices.tobytes()

    def toXML(self, writer, ttFont):
        formatstring, names, fixes = sstruct.getformat(postFormat)
        for name in names:
            value = getattr(self, name)
            writer.simpletag(name, value=value)
            writer.newline()
        if hasattr(self, 'mapping'):
            writer.begintag('psNames')
            writer.newline()
            writer.comment("This file uses unique glyph names based on the information\nfound in the 'post' table. Since these names might not be unique,\nwe have to invent artificial names in case of clashes. In order to\nbe able to retain the original information, we need a name to\nps name mapping for those cases where they differ. That's what\nyou see below.\n")
            writer.newline()
            items = sorted(self.mapping.items())
            for name, psName in items:
                writer.simpletag('psName', name=name, psName=psName)
                writer.newline()
            writer.endtag('psNames')
            writer.newline()
        if hasattr(self, 'extraNames'):
            writer.begintag('extraNames')
            writer.newline()
            writer.comment('following are the name that are not taken from the standard Mac glyph order')
            writer.newline()
            for name in self.extraNames:
                writer.simpletag('psName', name=name)
                writer.newline()
            writer.endtag('extraNames')
            writer.newline()
        if hasattr(self, 'data'):
            writer.begintag('hexdata')
            writer.newline()
            writer.dumphex(self.data)
            writer.endtag('hexdata')
            writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        if name not in ('psNames', 'extraNames', 'hexdata'):
            setattr(self, name, safeEval(attrs['value']))
        elif name == 'psNames':
            self.mapping = {}
            for element in content:
                if not isinstance(element, tuple):
                    continue
                name, attrs, content = element
                if name == 'psName':
                    self.mapping[attrs['name']] = attrs['psName']
        elif name == 'extraNames':
            self.extraNames = []
            for element in content:
                if not isinstance(element, tuple):
                    continue
                name, attrs, content = element
                if name == 'psName':
                    self.extraNames.append(attrs['name'])
        else:
            self.data = readHex(content)