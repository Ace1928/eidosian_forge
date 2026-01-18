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
class LCActionFlowable(ActionFlowable):
    locChanger = 1

    def wrap(self, availWidth, availHeight):
        """Should never be called."""
        raise NotImplementedError('%s.wrap should never be called' % self.__class__.__name__)

    def draw(self):
        """Should never be called."""
        raise NotImplementedError('%s.draw should never be called' % self.__class__.__name__)