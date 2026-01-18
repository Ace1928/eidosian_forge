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
class LIIndenter(DDIndenter):
    _IndenterAttrs = '_flowable _bullet _leftIndent _rightIndent width height spaceBefore spaceAfter'.split()

    def __init__(self, flowable, leftIndent=0, rightIndent=0, bullet=None, spaceBefore=None, spaceAfter=None):
        self._flowable = flowable
        self._bullet = bullet
        self._leftIndent = leftIndent
        self._rightIndent = rightIndent
        self.width = None
        self.height = None
        if spaceBefore is not None:
            self.spaceBefore = spaceBefore
        if spaceAfter is not None:
            self.spaceAfter = spaceAfter

    def split(self, aW, aH):
        S = self._flowable.split(aW - self._leftIndent - self._rightIndent, aH)
        return [LIIndenter(s, leftIndent=self._leftIndent, rightIndent=self._rightIndent, bullet=s is S[0] and self._bullet or None) for s in S]

    def drawOn(self, canv, x, y, _sW=0):
        if self._bullet:
            self._bullet.drawOn(self, canv, x, y, 0)
        self._flowable.drawOn(canv, x + self._leftIndent, y, max(0, _sW - self._leftIndent - self._rightIndent))