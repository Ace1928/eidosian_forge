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
class SetTopFlowables(NullDraw):
    _ZEROZSIZE = 1

    def __init__(self, F, show=False):
        self._F = F
        self._show = show

    def wrap(self, aW, aH):
        doc = getattr(getattr(self, 'canv', None), '_doctemplate', None)
        if doc:
            doc._topFlowables = self._F
            if self._show and self._F:
                doc.frame._generated_content = self._F
        return (0, 0)