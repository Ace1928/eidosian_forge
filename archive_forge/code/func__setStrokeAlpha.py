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
def _setStrokeAlpha(self, v):
    """
        Define the transparency/opacity of strokes. 0 is fully
        transparent, 1 is fully opaque.

        Note that calling this function will cause a version 1.4 PDF
        to be generated (rather than 1.3).
        """
    self._doc.ensureMinPdfVersion('transparency')
    self._extgstate.set(self, 'CA', v)