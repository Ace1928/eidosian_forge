import os
import sys
import tempfile
import time
from math import ceil, cos, pi, sin
from types import *
from . import pdfmetrics, pdfutils
from .pdfgeom import bezierArc
from .pdfutils import LINEEND  # this constant needed in both
class PDFPageCollection(PDFObject):
    """presumes PageList attribute set (list of integers)"""

    def __init__(self):
        self.PageList = []

    def save(self, file):
        lines = ['<<', '/Type /Pages', '/Count %d' % len(self.PageList), '/Kids [']
        for page in self.PageList:
            lines.append(str(page) + ' 0 R ')
        lines.append(']')
        lines.append('>>')
        text = LINEEND.join(lines)
        file.write(text + LINEEND)