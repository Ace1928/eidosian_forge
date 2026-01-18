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
class PDFSeparationCMYKColor:

    def __init__(self, cmyk):
        from reportlab.lib.colors import CMYKColor
        if not isinstance(cmyk, CMYKColor):
            raise ValueError('%s needs a CMYKColor argument' % self.__class__.__name__)
        elif not cmyk.spotName:
            raise ValueError('%s needs a CMYKColor argument with a spotName' % self.__class__.__name__)
        self.cmyk = cmyk

    def _makeFuncPS(self):
        """create the postscript code for the tint transfer function
        effectively this is tint*c, tint*y, ... tint*k"""
        R = [].append
        for i, v in enumerate(self.cmyk.cmyk()):
            v = float(v)
            if i == 3:
                if v == 0.0:
                    R('pop')
                    R('0.0')
                else:
                    R(str(v))
                    R('mul')
            else:
                if v == 0:
                    R('0.0')
                else:
                    R('dup')
                    R(str(v))
                    R('mul')
                R('exch')
        return '{%s}' % ' '.join(R.__self__)

    def value(self):
        return PDFArrayCompact((PDFName('Separation'), PDFName(self.cmyk.spotName), PDFName('DeviceCMYK'), PDFStream(dictionary=PDFDictionary(dict(FunctionType=4, Domain=PDFArrayCompact((0, 1)), Range=PDFArrayCompact((0, 1, 0, 1, 0, 1, 0, 1)))), content=self._makeFuncPS(), filters=None)))