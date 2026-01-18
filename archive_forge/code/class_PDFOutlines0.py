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
class PDFOutlines0(PDFObject):
    __Comment__ = 'TEST OUTLINE!'
    text = DUMMYOUTLINE.replace('\n', '\n')
    __RefOnly__ = 1

    def format(self, document):
        return pdfdocEnc(self.text)