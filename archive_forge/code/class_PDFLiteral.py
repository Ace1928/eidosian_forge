import os
import sys
import tempfile
import time
from math import ceil, cos, pi, sin
from types import *
from . import pdfmetrics, pdfutils
from .pdfgeom import bezierArc
from .pdfutils import LINEEND  # this constant needed in both
class PDFLiteral(PDFObject):
    """ a ready-made one you wish to quote"""

    def __init__(self, text):
        self.text = text

    def save(self, file):
        file.write(self.text + LINEEND)