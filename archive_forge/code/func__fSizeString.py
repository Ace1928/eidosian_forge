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
def _fSizeString(f):
    w = getattr(f, 'width', None)
    if w is None:
        w = getattr(f, '_width', None)
    h = getattr(f, 'height', None)
    if h is None:
        h = getattr(f, '_height', None)
    if hasattr(f, '_culprit'):
        c = ', %s, ' % f._culprit()
    else:
        c = ''
    if w is not None or h is not None:
        if w is None:
            w = '???'
        if h is None:
            h = '???'
        return '(%s x %s)%s' % (w, h, c)
    return ''