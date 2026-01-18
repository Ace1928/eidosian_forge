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
class PDFCatalog(PDFObject):
    __Comment__ = 'Document Root'
    __RefOnly__ = 1
    __Defaults__ = {'Type': PDFName('Catalog'), 'PageMode': PDFName('UseNone'), 'Lang': None}
    __NoDefault__ = '\n        Dests Outlines Pages Threads AcroForm Names OpenAction PageMode URI\n        ViewerPreferences PageLabels PageLayout JavaScript StructTreeRoot SpiderInfo\n        MarkInfo Metadata Tabs'.split()
    __Refs__ = __NoDefault__

    def format(self, document):
        self.check_format(document)
        defaults = self.__Defaults__
        Refs = self.__Refs__
        D = {}
        for k, v in defaults.items():
            v = getattr(self, k, v)
            if v is not None:
                D[k] = v
        for k in self.__NoDefault__:
            v = getattr(self, k, None)
            if v is not None:
                D[k] = v
        for k in Refs:
            if k in D:
                D[k] = document.Reference(D[k])
        dict = PDFDictionary(D)
        return format(dict, document)

    def showOutline(self):
        self.setPageMode('UseOutlines')

    def showFullScreen(self):
        self.setPageMode('FullScreen')

    def setPageLayout(self, layout):
        if layout:
            self.PageLayout = PDFName(layout)

    def setPageMode(self, mode):
        if mode:
            self.PageMode = PDFName(mode)

    def check_format(self, document):
        """for use in subclasses"""
        pass