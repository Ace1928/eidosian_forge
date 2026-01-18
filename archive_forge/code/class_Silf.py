from fontTools.misc import sstruct
from fontTools.misc.fixedTools import floatToFixedToStr
from fontTools.misc.textTools import byteord, safeEval
from . import DefaultTable
from . import grUtils
from array import array
from functools import reduce
import struct, re, sys
class Silf(object):
    """A particular Silf subtable"""

    def __init__(self):
        self.passes = []
        self.scriptTags = []
        self.critFeatures = []
        self.jLevels = []
        self.pMap = {}

    def decompile(self, data, ttFont, version=2.0):
        if version >= 3.0:
            _, data = sstruct.unpack2(Silf_part1_format_v3, data, self)
            self.ruleVersion = float(floatToFixedToStr(self.ruleVersion, precisionBits=16))
        _, data = sstruct.unpack2(Silf_part1_format, data, self)
        for jlevel in range(self.numJLevels):
            j, data = sstruct.unpack2(Silf_justify_format, data, _Object())
            self.jLevels.append(j)
        _, data = sstruct.unpack2(Silf_part2_format, data, self)
        if self.numCritFeatures:
            self.critFeatures = struct.unpack_from('>%dH' % self.numCritFeatures, data)
        data = data[self.numCritFeatures * 2 + 1:]
        numScriptTag, = struct.unpack_from('B', data)
        if numScriptTag:
            self.scriptTags = [struct.unpack('4s', data[x:x + 4])[0].decode('ascii') for x in range(1, 1 + 4 * numScriptTag, 4)]
        data = data[1 + 4 * numScriptTag:]
        self.lbGID, = struct.unpack('>H', data[:2])
        if self.numPasses:
            self.oPasses = struct.unpack('>%dL' % (self.numPasses + 1), data[2:6 + 4 * self.numPasses])
        data = data[6 + 4 * self.numPasses:]
        numPseudo, = struct.unpack('>H', data[:2])
        for i in range(numPseudo):
            if version >= 3.0:
                pseudo = sstruct.unpack(Silf_pseudomap_format, data[8 + 6 * i:14 + 6 * i], _Object())
            else:
                pseudo = sstruct.unpack(Silf_pseudomap_format_h, data[8 + 4 * i:12 + 4 * i], _Object())
            self.pMap[pseudo.unicode] = ttFont.getGlyphName(pseudo.nPseudo)
        data = data[8 + 6 * numPseudo:]
        currpos = sstruct.calcsize(Silf_part1_format) + sstruct.calcsize(Silf_justify_format) * self.numJLevels + sstruct.calcsize(Silf_part2_format) + 2 * self.numCritFeatures + 1 + 1 + 4 * numScriptTag + 6 + 4 * self.numPasses + 8 + 6 * numPseudo
        if version >= 3.0:
            currpos += sstruct.calcsize(Silf_part1_format_v3)
        self.classes = Classes()
        self.classes.decompile(data, ttFont, version)
        for i in range(self.numPasses):
            p = Pass()
            self.passes.append(p)
            p.decompile(data[self.oPasses[i] - currpos:self.oPasses[i + 1] - currpos], ttFont, version)

    def compile(self, ttFont, version=2.0):
        self.numPasses = len(self.passes)
        self.numJLevels = len(self.jLevels)
        self.numCritFeatures = len(self.critFeatures)
        numPseudo = len(self.pMap)
        data = b''
        if version >= 3.0:
            hdroffset = sstruct.calcsize(Silf_part1_format_v3)
        else:
            hdroffset = 0
        data += sstruct.pack(Silf_part1_format, self)
        for j in self.jLevels:
            data += sstruct.pack(Silf_justify_format, j)
        data += sstruct.pack(Silf_part2_format, self)
        if self.numCritFeatures:
            data += struct.pack('>%dH' % self.numCritFeaturs, *self.critFeatures)
        data += struct.pack('BB', 0, len(self.scriptTags))
        if len(self.scriptTags):
            tdata = [struct.pack('4s', x.encode('ascii')) for x in self.scriptTags]
            data += b''.join(tdata)
        data += struct.pack('>H', self.lbGID)
        self.passOffset = len(data)
        data1 = grUtils.bininfo(numPseudo, 6)
        currpos = hdroffset + len(data) + 4 * (self.numPasses + 1)
        self.pseudosOffset = currpos + len(data1)
        for u, p in sorted(self.pMap.items()):
            data1 += struct.pack('>LH' if version >= 3.0 else '>HH', u, ttFont.getGlyphID(p))
        data1 += self.classes.compile(ttFont, version)
        currpos += len(data1)
        data2 = b''
        datao = b''
        for i, p in enumerate(self.passes):
            base = currpos + len(data2)
            datao += struct.pack('>L', base)
            data2 += p.compile(ttFont, base, version)
        datao += struct.pack('>L', currpos + len(data2))
        if version >= 3.0:
            data3 = sstruct.pack(Silf_part1_format_v3, self)
        else:
            data3 = b''
        return data3 + data + datao + data1 + data2

    def toXML(self, writer, ttFont, version=2.0):
        if version >= 3.0:
            writer.simpletag('version', ruleVersion=self.ruleVersion)
            writer.newline()
        writesimple('info', self, writer, *attrs_info)
        writesimple('passindexes', self, writer, *attrs_passindexes)
        writesimple('contexts', self, writer, *attrs_contexts)
        writesimple('attributes', self, writer, *attrs_attributes)
        if len(self.jLevels):
            writer.begintag('justifications')
            writer.newline()
            jformat, jnames, jfixes = sstruct.getformat(Silf_justify_format)
            for i, j in enumerate(self.jLevels):
                attrs = dict([(k, getattr(j, k)) for k in jnames])
                writer.simpletag('justify', **attrs)
                writer.newline()
            writer.endtag('justifications')
            writer.newline()
        if len(self.critFeatures):
            writer.begintag('critFeatures')
            writer.newline()
            writer.write(' '.join(map(str, self.critFeatures)))
            writer.newline()
            writer.endtag('critFeatures')
            writer.newline()
        if len(self.scriptTags):
            writer.begintag('scriptTags')
            writer.newline()
            writer.write(' '.join(self.scriptTags))
            writer.newline()
            writer.endtag('scriptTags')
            writer.newline()
        if self.pMap:
            writer.begintag('pseudoMap')
            writer.newline()
            for k, v in sorted(self.pMap.items()):
                writer.simpletag('pseudo', unicode=hex(k), pseudo=v)
                writer.newline()
            writer.endtag('pseudoMap')
            writer.newline()
        self.classes.toXML(writer, ttFont, version)
        if len(self.passes):
            writer.begintag('passes')
            writer.newline()
            for i, p in enumerate(self.passes):
                writer.begintag('pass', _index=i)
                writer.newline()
                p.toXML(writer, ttFont, version)
                writer.endtag('pass')
                writer.newline()
            writer.endtag('passes')
            writer.newline()

    def fromXML(self, name, attrs, content, ttFont, version=2.0):
        if name == 'version':
            self.ruleVersion = float(safeEval(attrs.get('ruleVersion', '0')))
        if name == 'info':
            getSimple(self, attrs, *attrs_info)
        elif name == 'passindexes':
            getSimple(self, attrs, *attrs_passindexes)
        elif name == 'contexts':
            getSimple(self, attrs, *attrs_contexts)
        elif name == 'attributes':
            getSimple(self, attrs, *attrs_attributes)
        elif name == 'justifications':
            for element in content:
                if not isinstance(element, tuple):
                    continue
                tag, attrs, subcontent = element
                if tag == 'justify':
                    j = _Object()
                    for k, v in attrs.items():
                        setattr(j, k, int(v))
                    self.jLevels.append(j)
        elif name == 'critFeatures':
            self.critFeatures = []
            element = content_string(content)
            self.critFeatures.extend(map(int, element.split()))
        elif name == 'scriptTags':
            self.scriptTags = []
            element = content_string(content)
            for n in element.split():
                self.scriptTags.append(n)
        elif name == 'pseudoMap':
            self.pMap = {}
            for element in content:
                if not isinstance(element, tuple):
                    continue
                tag, attrs, subcontent = element
                if tag == 'pseudo':
                    k = int(attrs['unicode'], 16)
                    v = attrs['pseudo']
                self.pMap[k] = v
        elif name == 'classes':
            self.classes = Classes()
            for element in content:
                if not isinstance(element, tuple):
                    continue
                tag, attrs, subcontent = element
                self.classes.fromXML(tag, attrs, subcontent, ttFont, version)
        elif name == 'passes':
            for element in content:
                if not isinstance(element, tuple):
                    continue
                tag, attrs, subcontent = element
                if tag == 'pass':
                    p = Pass()
                    for e in subcontent:
                        if not isinstance(e, tuple):
                            continue
                        p.fromXML(e[0], e[1], e[2], ttFont, version)
                    self.passes.append(p)