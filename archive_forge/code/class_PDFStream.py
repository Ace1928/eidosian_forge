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
class PDFStream(PDFObject):
    """set dictionary elements explicitly stream.dictionary[name]=value"""
    __RefOnly__ = 1

    def __init__(self, dictionary=None, content=None, filters=None):
        if dictionary is None:
            dictionary = PDFDictionary()
        self.dictionary = dictionary
        self.content = content
        self.filters = filters

    def format(self, document):
        dictionary = self.dictionary
        dictionary = PDFDictionary(dictionary.dict.copy())
        content = self.content
        filters = self.filters
        if self.content is None:
            raise ValueError('stream content not set')
        if filters is None:
            filters = document.defaultStreamFilters
        if filters is not None and 'Filter' not in dictionary.dict:
            rf = list(filters)
            rf.reverse()
            fnames = []
            for f in rf:
                content = f.encode(content)
                fnames.insert(0, PDFName(f.pdfname))
            dictionary['Filter'] = PDFArray(fnames)
        content = document.encrypt.encode(content)
        fc = format(content, document)
        dictionary['Length'] = len(content)
        fd = format(dictionary, document)
        return fd + b'\nstream\n' + fc + b'endstream\n'