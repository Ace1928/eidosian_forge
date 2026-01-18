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
class _AbsRect(NullDraw):
    _ZEROSIZE = 1
    _SPACETRANSFER = True

    def __init__(self, x, y, width, height, strokeWidth=0, strokeColor=None, fillColor=None, strokeDashArray=None):
        self._x = x
        self._y = y
        self._width = width
        self._height = height
        self._strokeColor = strokeColor
        self._fillColor = fillColor
        self._strokeWidth = strokeWidth
        self._strokeDashArray = strokeDashArray

    def wrap(self, availWidth, availHeight):
        return (0, 0)

    def drawOn(self, canv, x, y, _sW=0):
        if self._width > _FUZZ and self._height > _FUZZ:
            st = self._strokeColor and self._strokeWidth is not None and (self._strokeWidth >= 0)
            if st or self._fillColor:
                canv.saveState()
                if st:
                    canv.setStrokeColor(self._strokeColor)
                    canv.setLineWidth(self._strokeWidth)
                if self._fillColor:
                    canv.setFillColor(self._fillColor)
                canv.rect(self._x, self._y, self._width, self._height, stroke=1 if st else 0, fill=1 if self._fillColor else 0)
                canv.restoreState()