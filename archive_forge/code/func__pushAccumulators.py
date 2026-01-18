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
def _pushAccumulators(self):
    """when you enter a form, save accumulator info not related to the form for page (if any)"""
    saved = (self._code, self._formsinuse, self._annotationrefs, self._formData, self._colorsUsed, self._shadingUsed)
    self._codeStack.append(saved)
    self._code = []
    self._currentPageHasImages = 1
    self._formsinuse = []
    self._annotationrefs = []
    self._formData = None
    self._colorsUsed = {}
    self._shadingUsed = {}