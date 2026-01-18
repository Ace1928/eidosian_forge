import os, sys, encodings
from reportlab.pdfbase import _fontdata
from reportlab.lib.logger import warnOnce
from reportlab.lib.utils import rl_isfile, rl_glob, rl_isdir, open_and_read, open_and_readlines, findInPaths, isSeq, isStr
from reportlab.rl_config import defaultEncoding, T1SearchPath
from reportlab.lib.rl_accel import unicode2T1, instanceStringWidthT1
from reportlab.pdfbase import rl_codecs
from reportlab.rl_config import register_reset
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