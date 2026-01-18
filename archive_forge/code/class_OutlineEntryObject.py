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
class OutlineEntryObject(PDFObject):
    """an entry in an outline"""
    Title = Dest = Parent = Prev = Next = First = Last = Count = None

    def format(self, document):
        D = {}
        D['Title'] = PDFString(self.Title)
        D['Parent'] = self.Parent
        D['Dest'] = self.Dest
        for n in ('Prev', 'Next', 'First', 'Last', 'Count'):
            v = getattr(self, n)
            if v is not None:
                D[n] = v
        PD = PDFDictionary(D)
        return PD.format(document)