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
class PTCycle(list):

    def __new__(cls, *args, **kwds):
        self = list.__new__(cls, *args, **kwds)
        self._restart = 0
        self._idx = 0
        return self

    @property
    def next_value(self):
        v = self[self._idx]
        self._idx += 1
        if self._idx >= len(self):
            self._idx = self._restart
        return v

    @property
    def peek(self):
        return self[self._idx]