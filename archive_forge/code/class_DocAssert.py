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
class DocAssert(DocPara):

    def __init__(self, cond, format=None):
        Flowable.__init__(self)
        self.expr = cond
        self.format = format

    def funcWrap(self, aW, aH):
        self._cond = DocPara.funcWrap(self, aW, aH)
        return self._cond

    def wrap(self, aW, aH):
        value = self.get_value(aW, aH)
        if not bool(self._cond):
            raise AssertionError(value)
        return (0, 0)