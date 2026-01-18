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
class PDFRectangle(PDFObject):

    def __init__(self, llx, lly, urx, ury):
        self.llx, self.lly, self.ulx, self.ury = (llx, lly, urx, ury)

    def format(self, document):
        A = PDFArray([self.llx, self.lly, self.ulx, self.ury])
        return format(A, document)