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
def _hAlignAdjust(self, x, sW=0):
    if sW and hasattr(self, 'hAlign'):
        a = self.hAlign
        if a in ('CENTER', 'CENTRE', TA_CENTER):
            x += 0.5 * sW
        elif a in ('RIGHT', TA_RIGHT):
            x += sW
        elif a not in ('LEFT', TA_LEFT):
            raise ValueError('Bad hAlign value ' + str(a))
    return x