import os
from copy import deepcopy, copy
from reportlab.lib.colors import gray, lightgrey
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.lib.styles import _baseFontName
from reportlab.lib.utils import strTypes, rl_safe_exec, annotateException
from reportlab.lib.abag import ABag
from reportlab.pdfbase import pdfutils
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.rl_config import _FUZZ, overlapAttachedSpace, ignoreContainerActions, listWrapOnFakeWidth
from reportlab.lib.sequencer import _type2formatter
from reportlab.lib.styles import ListStyle
def _setup_inner(self):
    width = self._width
    height = self._height
    kind = self._kind
    img = self._img
    if img:
        self.imageWidth, self.imageHeight = img.getSize()
        if self._dpi and hasattr(img, '_image'):
            self._dpi = img._image.info.get('dpi', (72, 72))
    elif self._drawing:
        self.imageWidth, self.imageHeight = (self._drawing.width, self._drawing.height)
        self._dpi = False
    self._dpiAdjust()
    if self._lazy >= 2:
        del self._img
    if kind in ['direct', 'absolute']:
        self.drawWidth = width or self.imageWidth
        self.drawHeight = height or self.imageHeight
    elif kind in ['percentage', '%']:
        self.drawWidth = self.imageWidth * width * 0.01
        self.drawHeight = self.imageHeight * height * 0.01
    elif kind in ['bound', 'proportional']:
        factor = min(float(width) / self.imageWidth, float(height) / self.imageHeight)
        self.drawWidth = self.imageWidth * factor
        self.drawHeight = self.imageHeight * factor