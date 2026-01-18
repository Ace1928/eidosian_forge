from reportlab.platypus.flowables import *
from reportlab.platypus.flowables import _ContainerSpace
from reportlab.lib.units import inch
from reportlab.platypus.paragraph import Paragraph
from reportlab.platypus.frames import Frame
from reportlab.rl_config import defaultPageSize, verbose
import reportlab.lib.sequencer
from reportlab.pdfgen import canvas
from reportlab.lib.utils import isSeq, encode_label, decode_label, annotateException, strTypes
import sys
import logging
class FrameActionFlowable(Flowable):
    _fixedWidth = _fixedHeight = 1

    def __init__(self, *arg, **kw):
        raise NotImplementedError('%s.__init__ should never be called for abstract Class' % self.__class__.__name__)

    def frameAction(self, frame):
        raise NotImplementedError('%s.frameAction should never be called for abstract Class' % self.__class__.__name__)