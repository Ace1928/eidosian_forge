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
def _isbalanced(s):
    """test whether a string is balanced in parens"""
    s = _re_cleanparens.sub('', s)
    n = 0
    for c in s:
        if c == '(':
            n += 1
        else:
            n -= 1
            if n < 0:
                return 0
    return not n and 1 or 0