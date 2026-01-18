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
def _calc(self):
    self._rightMargin = self.pagesize[0] - self.rightMargin
    self._topMargin = self.pagesize[1] - self.topMargin
    self.width = self._rightMargin - self.leftMargin
    self.height = self._topMargin - self.bottomMargin