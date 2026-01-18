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
def ShadingDict(self, **kw):
    d = {}
    d.update(kw)
    for name in self.required:
        if name not in d:
            raise ValueError('keyword argument %s missing' % name)
    permitted = self.permitted
    for name in d.keys():
        if name not in permitted:
            raise ValueError('bad annotation dictionary name %s' % name)
    return PDFDictionary(d)