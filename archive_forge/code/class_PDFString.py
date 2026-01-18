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
class PDFString(PDFObject):
    unicodeEncValid = False

    def __init__(self, s, escape=1, enc='auto'):
        """s can be unicode/utf8 or a PDFString
        if escape is true then the output will be passed through escape
        if enc is raw then bytes will be left alone
        if enc is auto we'll try and automatically adapt to utf_16_be/utf_16_le if the
        effective string is not entirely in pdfdoc
        if self.unicodeEncValid unicode will use the specifed encoding
        """
        if isinstance(s, PDFString):
            self.s = s.s
            self.escape = s.escape
            self.enc = s.enc
        else:
            self.s = s
            self.escape = escape
            self.enc = enc

    def format(self, document):
        s = self.s
        enc = getattr(self, 'enc', 'auto')
        if isBytes(s):
            if enc == 'auto':
                try:
                    if s.startswith(codecs.BOM_UTF16_BE):
                        u = s.decode('utf_16_be')
                    elif s.startswith(codecs.BOM_UTF16_LE):
                        u = s.decode('utf_16_le')
                    else:
                        u = s.decode('utf8')
                    if _checkPdfdoc(u):
                        s = u.encode('pdfdoc')
                    else:
                        s = codecs.BOM_UTF16_BE + u.encode('utf_16_be')
                except:
                    try:
                        s.decode('pdfdoc')
                    except:
                        stderr.write('Error in %s' % (repr(s),))
                        raise
        elif isUnicode(s):
            if enc == 'auto':
                if _checkPdfdoc(s):
                    s = s.encode('pdfdoc')
                else:
                    s = codecs.BOM_UTF16_BE + s.encode('utf_16_be')
            elif self.unicodeEncValid:
                s = s.encode(self.enc)
            else:
                s = codecs.BOM_UTF16_BE + s.encode('utf_16_be')
        else:
            raise ValueError('PDFString argument must be str/unicode not %s' % type(s))
        escape = getattr(self, 'escape', 1)
        if not isinstance(document.encrypt, NoEncryption):
            s = document.encrypt.encode(s)
            escape = 1
        if escape:
            try:
                es = '(%s)' % escapePDF(s)
            except:
                raise ValueError('cannot escape %s %s' % (s, repr(s)))
            if escape & 2:
                es = es.replace('\\012', '\n')
            if escape & 4 and _isbalanced(es):
                es = es.replace('\\(', '(').replace('\\)', ')')
            return pdfdocEnc(es)
        else:
            return b'(' + s + b')'

    def __str__(self):
        return '(%s)' % escapePDF(self.s)