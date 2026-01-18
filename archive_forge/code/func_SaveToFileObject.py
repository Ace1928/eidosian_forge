import os
import sys
import tempfile
import time
from math import ceil, cos, pi, sin
from types import *
from . import pdfmetrics, pdfutils
from .pdfgeom import bezierArc
from .pdfutils import LINEEND  # this constant needed in both
def SaveToFileObject(self, fileobj):
    """Open a file, and ask each object in turn to write itself to
        the file.  Keep track of the file position at each point for
        use in the index at the end"""
    f = fileobj
    i = 1
    self.xref = []
    f.write('%PDF-1.2' + LINEEND)
    f.write('%�춾' + LINEEND)
    for obj in self.objects:
        pos = f.tell()
        self.xref.append(pos)
        f.write(str(i) + ' 0 obj' + LINEEND)
        obj.save(f)
        f.write('endobj' + LINEEND)
        i = i + 1
    self.writeXref(f)
    self.writeTrailer(f)
    f.write('%%EOF')
    if os.name == 'mac':
        import macfs
        try:
            macfs.FSSpec(filename).SetCreatorType('CARO', 'PDF ')
        except Exception:
            pass