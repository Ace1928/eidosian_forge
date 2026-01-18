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
def _makeLIIndenter(self, flowable, bullet, params=None):
    if params:
        leftIndent = params.get('leftIndent', self._leftIndent)
        rightIndent = params.get('rightIndent', self._rightIndent)
        spaceBefore = params.get('spaceBefore', None)
        spaceAfter = params.get('spaceAfter', None)
        return LIIndenter(flowable, leftIndent, rightIndent, bullet, spaceBefore=spaceBefore, spaceAfter=spaceAfter)
    else:
        return LIIndenter(flowable, self._leftIndent, self._rightIndent, bullet)