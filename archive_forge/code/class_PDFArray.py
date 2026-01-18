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
class PDFArray(PDFObject):
    multiline = True

    def __init__(self, sequence):
        self.sequence = list(sequence)

    def References(self, document):
        """make all objects in sequence references"""
        self.sequence = list(map(document.Reference, self.sequence))

    def format(self, document, IND=b'\n '):
        L = [format(e, document) for e in self.sequence]
        if self.multiline and rl_config.pdfMultiLine or self.multiline == 'forced':
            L = IND.join(L)
        else:
            n = len(L)
            if n > 10:
                t = L.insert
                for i in reversed(range(10, n, 10)):
                    t(i, b'\n ')
                L = b' '.join(L)
            else:
                L = b' '.join(L)
        return b'[ ' + L + b' ]'