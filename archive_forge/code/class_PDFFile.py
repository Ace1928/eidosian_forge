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
class PDFFile(PDFObject):

    def __init__(self, pdfVersion=PDF_VERSION_DEFAULT):
        self.strings = []
        self.write = self.strings.append
        self.offset = 0
        self.add(pdfdocEnc('%%PDF-%s.%s' % pdfVersion) + b'\n%\x93\x8c\x8b\x9e ReportLab Generated PDF document http://www.reportlab.com\n')

    def closeOrReset(self):
        pass

    def add(self, s):
        """should be constructed as late as possible, return position where placed"""
        s = pdfdocEnc(s)
        result = self.offset
        self.offset = result + len(s)
        self.write(s)
        return result

    def format(self, document):
        return b''.join(self.strings)