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
class CallerMacro(Flowable):
    """
    like Macro, but with callable command(s)
    drawCallable(self)
    wrapCallable(self,aW,aH)
    """

    def __init__(self, drawCallable=None, wrapCallable=None):
        self._drawCallable = drawCallable or _nullCallable
        self._wrapCallable = wrapCallable or _nullCallable

    def __repr__(self):
        return 'CallerMacro(%r,%r)' % (self._drawCallable, self._wrapCallable)

    def wrap(self, aW, aH):
        self._wrapCallable(self, aW, aH)
        return (0, 0)

    def draw(self):
        self._drawCallable(self)