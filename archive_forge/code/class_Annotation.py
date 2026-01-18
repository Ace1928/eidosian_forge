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
class Annotation(PDFObject):
    """superclass for all annotations."""
    defaults = [('Type', PDFName('Annot'))]
    required = ('Type', 'Rect', 'Contents', 'Subtype')
    permitted = required + ('Border', 'C', 'T', 'M', 'F', 'H', 'BS', 'AA', 'AS', 'Popup', 'P', 'AP')

    def cvtdict(self, d, escape=1):
        """transform dict args from python form to pdf string rep as needed"""
        Rect = d['Rect']
        if not isStr(Rect):
            d['Rect'] = PDFArray(Rect)
        d['Contents'] = PDFString(d['Contents'], escape)
        return d

    def AnnotationDict(self, **kw):
        if 'escape' in kw:
            escape = kw['escape']
            del kw['escape']
        else:
            escape = 1
        d = {}
        for name, val in self.defaults:
            d[name] = val
        d.update(kw)
        for name in self.required:
            if name not in d:
                raise ValueError('keyword argument %s missing' % name)
        d = self.cvtdict(d, escape=escape)
        permitted = self.permitted
        for name in d.keys():
            if name not in permitted:
                raise ValueError('%s bad annotation dictionary name %s' % (self.__class__.__name__, name))
        return PDFDictionary(d)

    def Dict(self):
        raise ValueError('DictString undefined for virtual superclass Annotation, must overload')

    def format(self, document):
        D = self.Dict()
        return D.format(document)