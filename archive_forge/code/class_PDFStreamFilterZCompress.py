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
class PDFStreamFilterZCompress:
    pdfname = 'FlateDecode'

    def encode(self, text):
        if isUnicode(text):
            text = text.encode('utf8')
        return zlib.compress(text)

    def decode(self, encoded):
        return zlib.decompress(encoded)