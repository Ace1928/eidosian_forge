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
class FreeTextAnnotation(Annotation):
    permitted = Annotation.permitted + ('DA',)

    def __init__(self, Rect, Contents, DA, **kw):
        self.Rect = Rect
        self.Contents = Contents
        self.DA = DA
        self.otherkw = kw

    def Dict(self):
        d = {}
        d.update(self.otherkw)
        d['Rect'] = self.Rect
        d['Contents'] = self.Contents
        d['DA'] = self.DA
        d['Subtype'] = '/FreeText'
        return self.AnnotationDict(**d)