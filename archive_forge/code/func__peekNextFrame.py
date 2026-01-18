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
def _peekNextFrame(self):
    """intended to be used by extreme flowables"""
    if hasattr(self, '_nextFrameIndex'):
        return self.pageTemplate.frames[self._nextFrameIndex]
    f = self.frame
    if hasattr(f, 'lastFrame') or f is self.pageTemplate.frames[-1]:
        if hasattr(self, '_nextPageTemplateCycle'):
            pageTemplate = self._nextPageTemplateCycle.peek
        elif hasattr(self, '_nextPageTemplateIndex'):
            pageTemplate = self.pageTemplates[self._nextPageTemplateIndex]
        elif self.pageTemplate.autoNextPageTemplate:
            pageTemplate = self._peekNextPageTemplate(self.pageTemplate.autoNextPageTemplate)
        else:
            pageTemplate = self.pageTemplate
        return pageTemplate.frames[0]
    else:
        return self.pageTemplate.frames[self.pageTemplate.frames.index(f) + 1]