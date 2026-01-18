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
def PDFName(data, lo=chr(33), hi=chr(126)):
    L = list(data)
    for i, c in enumerate(L):
        if c < lo or c > hi or c in '%()<>{}[]#':
            L[i] = '#' + hex(ord(c))[2:]
    return '/' + ''.join(L)