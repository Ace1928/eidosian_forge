import re
import hashlib
from string import digits
from math import sin, cos, tan, pi
from reportlab import rl_config
from reportlab.pdfbase import pdfdoc
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen  import pathobject
from reportlab.pdfgen.textobject import PDFTextObject, _PDFColorSetter
from reportlab.lib.colors import black, _chooseEnforceColorSpace, Color, CMYKColor, toColor
from reportlab.lib.utils import ImageReader, isSeq, isStr, isUnicode, _digester, asUnicode
from reportlab.lib.rl_accel import fp_str, escapePDF
from reportlab.lib.boxstuff import aspectRatioFix
def _absRect(self, rect, relative=0):
    if not rect:
        w, h = self._pagesize
        rect = (0, 0, w, h)
    elif relative:
        lx, ly, ux, uy = rect
        xll, yll = self.absolutePosition(lx, ly)
        xur, yur = self.absolutePosition(ux, uy)
        xul, yul = self.absolutePosition(lx, uy)
        xlr, ylr = self.absolutePosition(ux, ly)
        xs = (xll, xur, xul, xlr)
        ys = (yll, yur, yul, ylr)
        xmin, ymin = (min(xs), min(ys))
        xmax, ymax = (max(xs), max(ys))
        rect = (xmin, ymin, xmax, ymax)
    bw = self._getCmShift()
    if bw:
        rect = (rect[0] + bw, rect[1] + bw, rect[2] + bw, rect[3] + bw)
    return rect