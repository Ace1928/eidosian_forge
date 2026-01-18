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
def _restrictSize(self, aW, aH):
    if self.drawWidth > aW + _FUZZ or self.drawHeight > aH + _FUZZ:
        self._oldDrawSize = (self.drawWidth, self.drawHeight)
        factor = min(float(aW) / self.drawWidth, float(aH) / self.drawHeight)
        self.drawWidth *= factor
        self.drawHeight *= factor
    return (self.drawWidth, self.drawHeight)