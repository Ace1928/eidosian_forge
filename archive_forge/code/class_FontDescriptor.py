import sys, os, pickle
from hashlib import md5
from xml.sax.saxutils import quoteattr
from time import process_time as clock
from reportlab.lib.utils import asBytes, asNative as _asNative
from reportlab.lib.utils import rl_isdir, rl_isfile, rl_listdir, rl_getmtime
class FontDescriptor:
    """This is a short descriptive record about a font.

    typeCode should be a file extension e.g. ['ttf','ttc','otf','pfb','pfa']
    """

    def __init__(self):
        self.name = None
        self.fullName = None
        self.familyName = None
        self.styleName = None
        self.isBold = False
        self.isItalic = False
        self.isFixedPitch = False
        self.isSymbolic = False
        self.typeCode = None
        self.fileName = None
        self.metricsFileName = None
        self.timeModified = 0

    def __repr__(self):
        return 'FontDescriptor(%s)' % self.name

    def getTag(self):
        """Return an XML tag representation"""
        attrs = []
        for k, v in self.__dict__.items():
            if k not in ['timeModified']:
                if v:
                    attrs.append('%s=%s' % (k, quoteattr(str(v))))
        return '<font ' + ' '.join(attrs) + '/>'