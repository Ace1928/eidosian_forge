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
class DocWhile(DocIf):

    def __init__(self, cond, whileBlock):
        Flowable.__init__(self)
        self.expr = cond
        self.block = self.checkBlock(whileBlock)

    def wrap(self, aW, aH):
        if bool(self.funcWrap(aW, aH)):
            self.add_content(*list(self.block) + [self])
        return (0, 0)