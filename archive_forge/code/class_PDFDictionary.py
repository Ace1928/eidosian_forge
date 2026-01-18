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
class PDFDictionary(PDFObject):
    multiline = True

    def __init__(self, dict=None):
        """dict should be namestring to value eg "a": 122 NOT pdfname to value NOT "/a":122"""
        if dict is None:
            self.dict = {}
        else:
            self.dict = dict.copy()

    def __setitem__(self, name, value):
        self.dict[name] = value

    def __getitem__(self, a):
        return self.dict[a]

    def __contains__(self, a):
        return a in self.dict

    def Reference(self, name, document):
        self.dict[name] = document.Reference(self.dict[name])

    def format(self, document, IND=b'\n '):
        dict = self.dict
        try:
            keys = list(dict.keys())
        except:
            print(ascii(dict))
            raise
        if not isinstance(dict, OrderedDict):
            keys.sort()
        L = [format(PDFName(k), document) + b' ' + format(dict[k], document) for k in keys]
        if self.multiline and rl_config.pdfMultiLine or self.multiline == 'forced':
            L = IND.join(L)
        else:
            t = L.insert
            for i in reversed(range(6, len(L), 6)):
                t(i, b'\n ')
            L = b' '.join(L)
        return b'<<\n' + L + b'\n>>'

    def copy(self):
        return PDFDictionary(self.dict)

    def normalize(self):
        D = self.dict
        K = [k for k in D.keys() if k.startswith('/')]
        for k in K:
            D[k[1:]] = D.pop(k)