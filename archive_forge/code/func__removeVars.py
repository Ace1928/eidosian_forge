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
def _removeVars(self, lifetimes):
    """remove namespace variables for with lifetime in lifetimes"""
    LT = self._lifetimes
    NS = self._nameSpace
    for lifetime in lifetimes:
        for k in LT.setdefault(lifetime, []):
            try:
                del NS[k]
            except KeyError:
                pass
        del LT[lifetime]