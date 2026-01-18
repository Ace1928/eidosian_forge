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
class HRFlowable(Flowable):
    """Like the hr tag"""

    def __init__(self, width='80%', thickness=1, lineCap='round', color=lightgrey, spaceBefore=1, spaceAfter=1, hAlign='CENTER', vAlign='BOTTOM', dash=None):
        Flowable.__init__(self)
        self.width = width
        self.lineWidth = thickness
        self.lineCap = lineCap
        self.spaceBefore = spaceBefore
        self.spaceAfter = spaceAfter
        self.color = color
        self.hAlign = hAlign
        self.vAlign = vAlign
        self.dash = dash

    def __repr__(self):
        return 'HRFlowable(width=%s, height=%s)' % (self.width, self.height)

    def wrap(self, availWidth, availHeight):
        w = self.width
        if isinstance(w, strTypes):
            w = w.strip()
            if w.endswith('%'):
                w = availWidth * float(w[:-1]) * 0.01
            else:
                w = float(w)
        w = min(w, availWidth)
        self._width = w
        return (w, self.lineWidth)

    def draw(self):
        canv = self.canv
        canv.saveState()
        canv.setLineWidth(self.lineWidth)
        canv.setLineCap({'butt': 0, 'round': 1, 'square': 2}[self.lineCap.lower()])
        canv.setStrokeColor(self.color)
        if self.dash:
            canv.setDash(self.dash)
        canv.line(0, 0, self._width, self.height)
        canv.restoreState()