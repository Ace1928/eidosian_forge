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
def _endBuild(self):
    self._removeVars(('build', 'page', 'frame'))
    if self._hanging != [] and self._hanging[-1] is PageBegin:
        del self._hanging[-1]
        self.clean_hanging()
    else:
        self.clean_hanging()
        self.handle_pageBreak()
    if getattr(self, '_doSave', 1):
        self.canv.save()
    if self._onPage:
        self.canv.setPageCallBack(None)