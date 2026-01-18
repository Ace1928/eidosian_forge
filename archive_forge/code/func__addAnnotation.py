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
def _addAnnotation(self, annotation, name=None, addtopage=1):
    count = self._annotationCount = self._annotationCount + 1
    if not name:
        name = 'NUMBER' + repr(count)
    self._doc.addAnnotation(name, annotation)
    if addtopage:
        self._annotatePage(name)