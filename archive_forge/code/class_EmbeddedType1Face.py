import os, sys, encodings
from reportlab.pdfbase import _fontdata
from reportlab.lib.logger import warnOnce
from reportlab.lib.utils import rl_isfile, rl_glob, rl_isdir, open_and_read, open_and_readlines, findInPaths, isSeq, isStr
from reportlab.rl_config import defaultEncoding, T1SearchPath
from reportlab.lib.rl_accel import unicode2T1, instanceStringWidthT1
from reportlab.pdfbase import rl_codecs
from reportlab.rl_config import register_reset
class EmbeddedType1Face(TypeFace):
    """A Type 1 font other than one of the basic 14.

    Its glyph data will be embedded in the PDF file."""

    def __init__(self, afmFileName, pfbFileName):
        TypeFace.__init__(self, None)
        afmFileName = findInPaths(afmFileName, T1SearchPath)
        pfbFileName = findInPaths(pfbFileName, T1SearchPath)
        self.afmFileName = os.path.abspath(afmFileName)
        self.pfbFileName = os.path.abspath(pfbFileName)
        self.requiredEncoding = None
        self._loadGlyphs(pfbFileName)
        self._loadMetrics(afmFileName)

    def getFontFiles(self):
        return [self.afmFileName, self.pfbFileName]

    def _loadGlyphs(self, pfbFileName):
        """Loads in binary glyph data, and finds the four length
        measurements needed for the font descriptor"""
        pfbFileName = bruteForceSearchForFile(pfbFileName)
        assert rl_isfile(pfbFileName), 'file %s not found' % pfbFileName
        d = open_and_read(pfbFileName, 'b')
        s1, l1 = _pfbCheck(0, d, PFB_ASCII, pfbFileName)
        s2, l2 = _pfbCheck(l1, d, PFB_BINARY, pfbFileName)
        s3, l3 = _pfbCheck(l2, d, PFB_ASCII, pfbFileName)
        _pfbCheck(l3, d, PFB_EOF, pfbFileName)
        self._binaryData = d[s1:l1] + d[s2:l2] + d[s3:l3]
        self._length = len(self._binaryData)
        self._length1 = l1 - s1
        self._length2 = l2 - s2
        self._length3 = l3 - s3

    def _loadMetrics(self, afmFileName):
        """Loads in and parses font metrics"""
        afmFileName = bruteForceSearchForFile(afmFileName)
        topLevel, glyphData = parseAFMFile(afmFileName)
        self.name = topLevel['FontName']
        self.familyName = topLevel['FamilyName']
        self.ascent = topLevel.get('Ascender', 1000)
        self.descent = topLevel.get('Descender', 0)
        self.capHeight = topLevel.get('CapHeight', 1000)
        self.italicAngle = topLevel.get('ItalicAngle', 0)
        self.stemV = topLevel.get('stemV', 0)
        self.xHeight = topLevel.get('XHeight', 1000)
        strBbox = topLevel.get('FontBBox', [0, 0, 1000, 1000])
        tokens = strBbox.split()
        self.bbox = []
        for tok in tokens:
            self.bbox.append(int(tok))
        glyphWidths = {}
        for cid, width, name in glyphData:
            glyphWidths[name] = width
        self.glyphWidths = glyphWidths
        self.glyphNames = list(glyphWidths.keys())
        self.glyphNames.sort()
        if topLevel.get('EncodingScheme', None) == 'FontSpecific':
            global _postScriptNames2Unicode
            if _postScriptNames2Unicode is None:
                try:
                    from reportlab.pdfbase._glyphlist import _glyphname2unicode
                    _postScriptNames2Unicode = _glyphname2unicode
                    del _glyphname2unicode
                except:
                    _postScriptNames2Unicode = {}
                    raise ValueError('cannot import module reportlab.pdfbase._glyphlist module\nyou can obtain a version from here\nhttps://www.reportlab.com/ftp/_glyphlist.py\n')
            names = [None] * 256
            ex = {}
            rex = {}
            for code, width, name in glyphData:
                if 0 <= code <= 255:
                    names[code] = name
                    u = _postScriptNames2Unicode.get(name, None)
                    if u is not None:
                        rex[code] = u
                        ex[u] = code
            encName = encodings.normalize_encoding('rl-dynamic-%s-encoding' % self.name)
            rl_codecs.RL_Codecs.add_dynamic_codec(encName, ex, rex)
            self.requiredEncoding = encName
            enc = Encoding(encName, names)
            registerEncoding(enc)

    def addObjects(self, doc):
        """Add whatever needed to PDF file, and return a FontDescriptor reference"""
        from reportlab.pdfbase import pdfdoc
        fontFile = pdfdoc.PDFStream()
        fontFile.content = self._binaryData
        fontFile.dictionary['Length1'] = self._length1
        fontFile.dictionary['Length2'] = self._length2
        fontFile.dictionary['Length3'] = self._length3
        fontFileRef = doc.Reference(fontFile, 'fontFile:' + self.pfbFileName)
        fontDescriptor = pdfdoc.PDFDictionary({'Type': '/FontDescriptor', 'Ascent': self.ascent, 'CapHeight': self.capHeight, 'Descent': self.descent, 'Flags': 34, 'FontBBox': pdfdoc.PDFArray(self.bbox), 'FontName': pdfdoc.PDFName(self.name), 'ItalicAngle': self.italicAngle, 'StemV': self.stemV, 'XHeight': self.xHeight, 'FontFile': fontFileRef})
        fontDescriptorRef = doc.Reference(fontDescriptor, 'fontDescriptor:' + self.name)
        return fontDescriptorRef