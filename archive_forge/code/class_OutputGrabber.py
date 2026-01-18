import os
import sys
import tempfile
import time
from math import ceil, cos, pi, sin
from types import *
from . import pdfmetrics, pdfutils
from .pdfgeom import bezierArc
from .pdfutils import LINEEND  # this constant needed in both
class OutputGrabber:
    """At times we need to put something in the place of standard
    output.  This grabs stdout, keeps the data, and releases stdout
    when done.
    
    NOT working well enough!"""

    def __init__(self):
        self.oldoutput = sys.stdout
        sys.stdout = self
        self.closed = 0
        self.data = []

    def write(self, x):
        if not self.closed:
            self.data.append(x)

    def getData(self):
        return ' '.join(self.data)

    def close(self):
        sys.stdout = self.oldoutput
        self.closed = 1

    def __del__(self):
        if not self.closed:
            self.close()