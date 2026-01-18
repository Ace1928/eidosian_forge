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
class NextPageTemplate(ActionFlowable):
    locChanger = 1
    'When you get to the next page, use the template specified (change to two column, for example)  '

    def __init__(self, pt):
        ActionFlowable.__init__(self, ('nextPageTemplate', pt))