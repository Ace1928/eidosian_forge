import binascii, codecs, zlib
from collections import OrderedDict
from reportlab.pdfbase import pdfutils
from reportlab import rl_config
from reportlab.lib.utils import open_for_read, makeFileName, isSeq, isBytes, isUnicode, _digester, isStr, bytestr, annotateException, TimeStamp
from reportlab.lib.rl_accel import escapePDF, fp_str, asciiBase85Encode, asciiBase85Decode
from reportlab.pdfbase import pdfmetrics
from hashlib import md5
from sys import stderr
import re
def GetPDFData(self, canvas):
    for fnt in self.delayedFonts:
        fnt.addObjects(self)
    self.info.invariant = self.invariant
    self.info.digest(self.signature)
    self.Reference(self.Catalog)
    self.Reference(self.info)
    self.Outlines.prepare(self, canvas)
    if self.Outlines.ready < 0:
        self.Catalog.Outlines = None
    return self.format()