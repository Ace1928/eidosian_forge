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
class PDFRadialShading(PDFShading):
    required = PDFShading.required + ('Coords', 'Function')
    permitted = PDFShading.permitted + ('Coords', 'Domain', 'Function', 'Extend')

    def __init__(self, x0, y0, r0, x1, y1, r1, Function, ColorSpace, **kw):
        self.Coords = (x0, y0, r0, x1, y1, r1)
        self.Function = Function
        self.ColorSpace = ColorSpace
        self.otherkw = kw

    def Dict(self, document):
        d = {}
        d.update(self.otherkw)
        d['ShadingType'] = 3
        d['ColorSpace'] = PDFName(self.ColorSpace)
        d['Coords'] = PDFArrayCompact(self.Coords)
        d['Function'] = document.Reference(self.Function)
        return self.ShadingDict(**d)