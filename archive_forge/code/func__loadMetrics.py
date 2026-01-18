import os, sys, encodings
from reportlab.pdfbase import _fontdata
from reportlab.lib.logger import warnOnce
from reportlab.lib.utils import rl_isfile, rl_glob, rl_isdir, open_and_read, open_and_readlines, findInPaths, isSeq, isStr
from reportlab.rl_config import defaultEncoding, T1SearchPath
from reportlab.lib.rl_accel import unicode2T1, instanceStringWidthT1
from reportlab.pdfbase import rl_codecs
from reportlab.rl_config import register_reset
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