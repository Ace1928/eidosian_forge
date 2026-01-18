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
class PDFIndirectObject(PDFObject):
    __RefOnly__ = 1

    def __init__(self, name, content):
        self.name = name
        self.content = content

    def format(self, document):
        name = self.name
        n, v = document.idToObjectNumberAndVersion[name]
        document.encrypt.register(n, v)
        fcontent = format(self.content, document, toplevel=1)
        return pdfdocEnc('%s %s obj\n' % (n, v)) + fcontent + (b'' if fcontent.endswith(b'\n') else b'\n') + b'endobj\n'