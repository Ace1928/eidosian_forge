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
class HighlightAnnotation(Annotation):
    """
    HighlightAnnotation is an annotation that highlights the selected area.

    Rect is the mouseover area that will show the contents.

    QuadPoints is a list of points to highlight, you can have many groups of
    four QuadPoints to allow highlighting many lines.
    """
    permitted = Annotation.permitted + ('QuadPoints',)

    def __init__(self, Rect, Contents, QuadPoints, Color=[0.83, 0.89, 0.95], **kw):
        self.Rect = Rect
        self.Contents = Contents
        self.otherkw = kw
        self.QuadPoints = QuadPoints
        self.Color = Color

    def cvtdict(self, d, escape=1):
        """transform dict args from python form to pdf string rep as needed"""
        Rect = d['Rect']
        Quad = d['QuadPoints']
        Color = d['C']
        if not isinstance(Rect, str):
            d['Rect'] = PDFArray(Rect).format(d, IND=b' ')
        if not isinstance(Quad, str):
            d['QuadPoints'] = PDFArray(Quad).format(d, IND=b' ')
        if not isinstance(Color, str):
            d['C'] = PDFArray(Color).format(d, IND=b' ')
        d['Contents'] = PDFString(d['Contents'], escape)
        return d

    def Dict(self):
        d = {}
        d.update(self.otherkw)
        d['Rect'] = self.Rect
        d['Contents'] = self.Contents
        d['Subtype'] = '/Highlight'
        d['QuadPoints'] = self.QuadPoints
        d['C'] = self.Color
        return self.AnnotationDict(**d)