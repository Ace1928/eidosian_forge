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
def ID(self):
    """A unique fingerprint for the file (unless in invariant mode)"""
    if self._ID:
        return self._ID
    digest = self.signature.digest()
    doc = DummyDoc()
    IDs = PDFText(digest, enc='raw').format(doc)
    self._ID = b'\n[' + IDs + IDs + b']\n% ReportLab generated PDF document -- digest (http://www.reportlab.com)\n'
    return self._ID