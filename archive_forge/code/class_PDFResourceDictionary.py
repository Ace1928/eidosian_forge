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
class PDFResourceDictionary(PDFObject):
    """each element *could* be reset to a reference if desired"""

    def __init__(self, **kwds):
        for _ in self.dict_attributes:
            setattr(self, _, kwds.pop(_, {}))
        self.basicProcs()
        if 'ProcSet' in kwds:
            self.ProcSet = kwds.pop('ProcSet')
    stdprocs = [PDFName(s) for s in 'PDF Text ImageB ImageC ImageI'.split()]
    dict_attributes = ('ColorSpace', 'XObject', 'ExtGState', 'Font', 'Pattern', 'Properties', 'Shading')

    def allProcs(self):
        self.ProcSet = self.stdprocs

    def basicProcs(self):
        self.ProcSet = self.stdprocs[:2]

    def basicFonts(self):
        self.Font = PDFObjectReference(BasicFonts)

    def setColorSpace(self, colorsUsed):
        for c, s in colorsUsed.items():
            self.ColorSpace[s] = PDFObjectReference(c)

    def setShading(self, shadingUsed):
        for c, s in shadingUsed.items():
            self.Shading[s] = PDFObjectReference(c)

    def format(self, document):
        D = {}
        for dname in self.dict_attributes:
            v = getattr(self, dname)
            if isinstance(v, dict):
                if v:
                    dv = PDFDictionary(v)
                    D[dname] = dv
            else:
                D[dname] = v
        v = self.ProcSet
        dname = 'ProcSet'
        if isSeq(v):
            if v:
                dv = PDFArray(v)
                D[dname] = dv
        else:
            D[dname] = v
        DD = PDFDictionary(D)
        return format(DD, document)