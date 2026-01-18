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
class PDFCrossReferenceTable(PDFObject):

    def __init__(self):
        self.sections = []

    def addsection(self, firstentry, ids):
        section = PDFCrossReferenceSubsection(firstentry, ids)
        self.sections.append(section)

    def format(self, document):
        sections = self.sections
        if not sections:
            raise ValueError('no crossref sections')
        L = [b'xref\n']
        for s in self.sections:
            fs = format(s, document)
            L.append(fs)
        return pdfdocEnc(b''.join(L))