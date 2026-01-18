import os, sys, encodings
from reportlab.pdfbase import _fontdata
from reportlab.lib.logger import warnOnce
from reportlab.lib.utils import rl_isfile, rl_glob, rl_isdir, open_and_read, open_and_readlines, findInPaths, isSeq, isStr
from reportlab.rl_config import defaultEncoding, T1SearchPath
from reportlab.lib.rl_accel import unicode2T1, instanceStringWidthT1
from reportlab.pdfbase import rl_codecs
from reportlab.rl_config import register_reset
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