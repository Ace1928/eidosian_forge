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
class TAction(ActionFlowable):
    """a special Action flowable that sets stuff on the doc template T"""

    def __init__(self, bgs=[], F=[], f=None):
        Flowable.__init__(self)
        self.bgs = bgs
        self.F = F
        self.f = f

    def apply(self, doc, T=T):
        T.frames = self.F
        frame._frameBGs = self.bgs
        doc.handle_currentFrame(self.f.id)
        frame._frameBGs = self.bgs