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
def _checkPdfdoc(utext):
    """return true if no Pdfdoc encoding errors"""
    try:
        utext.encode('pdfdoc')
        return 1
    except UnicodeEncodeError as e:
        return 0