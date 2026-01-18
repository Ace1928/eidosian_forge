import math, sys, os, codecs, base64
from io import BytesIO, StringIO
from reportlab.pdfbase.pdfmetrics import stringWidth # for font info
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.utils import asNative
from reportlab.graphics.renderbase import getStateDelta, Renderer, renderScaledDrawing
from reportlab.graphics.shapes import STATE_DEFAULTS, Path, UserNode
from reportlab.graphics.shapes import * # (only for test0)
from reportlab import rl_config
from reportlab.lib.utils import RLString, isUnicode, isBytes
from reportlab.pdfgen.canvas import FILL_EVEN_ODD, FILL_NON_ZERO
from .renderPM import _getImage
from xml.dom import getDOMImplementation
class EncodedWriter(list):
    """
    EncodedWriter(encoding) assumes .write will be called with
    either unicode or utf8 encoded bytes. it will accumulate
    unicode
    """
    BOMS = {'utf-32': codecs.BOM_UTF32, 'utf-32-be': codecs.BOM_UTF32_BE, 'utf-32-le': codecs.BOM_UTF32_LE, 'utf-16': codecs.BOM_UTF16, 'utf-16-be': codecs.BOM_UTF16_BE, 'utf-16-le': codecs.BOM_UTF16_LE}

    def __init__(self, encoding, bom=False):
        list.__init__(self)
        self.encoding = encoding = codecs.lookup(encoding).name
        if bom and '16' in encoding or '32' in encoding:
            self.write(self.BOMS[encoding])

    def write(self, u):
        if isBytes(u):
            try:
                u = u.decode('utf-8')
            except:
                et, ev, tb = sys.exc_info()
                ev = str(ev)
                del et, tb
                raise ValueError("String %r not encoded as 'utf-8'\nerror=%s" % (u, ev))
        elif not isUnicode(u):
            raise ValueError("EncodedWriter.write(%s) argument should be 'utf-8' bytes or str" % ascii(u))
        self.append(u)

    def getvalue(self):
        r = ''.join(self)
        del self[:]
        return r