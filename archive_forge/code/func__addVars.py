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
def _addVars(self, vars, lifetime):
    """add namespace variables to lifetimes lists"""
    LT = self._lifetimes
    for var in vars:
        for v in LT.values():
            if var in v:
                v.remove(var)
        LT.setdefault(lifetime, set([])).add(var)