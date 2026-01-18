import os
import marshal
import time
from hashlib import md5
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase._cidfontdata import allowedTypeFaces, allowedEncodings, CIDFontInfo, \
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase import pdfdoc
from reportlab.lib.rl_accel import escapePDF
from reportlab.rl_config import CMapSearchPath
from reportlab.lib.utils import isSeq, isBytes
def _extractDictInfo(self, name):
    try:
        fontDict = CIDFontInfo[name]
    except KeyError:
        raise KeyError("Unable to find information on CID typeface '%s'" % name + 'Only the following font names work:' + repr(allowedTypeFaces))
    descFont = fontDict['DescendantFonts'][0]
    self.ascent = descFont['FontDescriptor']['Ascent']
    self.descent = descFont['FontDescriptor']['Descent']
    self._defaultWidth = descFont['DW']
    self._explicitWidths = self._expandWidths(descFont['W'])