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
class onDrawStr(str):

    def __new__(cls, value, onDraw, label, kind=None):
        self = str.__new__(cls, value)
        self.onDraw = onDraw
        self.kind = kind
        self.label = label
        return self

    def __getnewargs__(self):
        return (str(self), self.onDraw, self.label, self.kind)