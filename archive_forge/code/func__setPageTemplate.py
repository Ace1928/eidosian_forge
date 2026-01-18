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
def _setPageTemplate(self):
    if hasattr(self, '_nextPageTemplateCycle'):
        self.pageTemplate = self._nextPageTemplateCycle.next_value
    elif hasattr(self, '_nextPageTemplateIndex'):
        self.pageTemplate = self.pageTemplates[self._nextPageTemplateIndex]
        del self._nextPageTemplateIndex
    elif self.pageTemplate.autoNextPageTemplate:
        self.handle_nextPageTemplate(self.pageTemplate.autoNextPageTemplate)
        self.pageTemplate = self.pageTemplates[self._nextPageTemplateIndex]