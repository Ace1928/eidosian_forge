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
class PageAccumulator:
    """gadget to accumulate information in a page
    and then allow it to be interrogated at the end
    of the page"""
    _count = 0

    def __init__(self, name=None):
        if name is None:
            name = self.__class__.__name__ + str(self.__class__._count)
            self.__class__._count += 1
        self.name = name
        self.data = []

    def reset(self):
        self.data[:] = []

    def add(self, *args):
        self.data.append(args)

    def onDrawText(self, *args):
        return '<onDraw name="%s" label="%s" />' % (self.name, encode_label(args))

    def __call__(self, canv, kind, label):
        self.add(*decode_label(label))

    def attachToPageTemplate(self, pt):
        if pt.onPage:

            def onPage(canv, doc, oop=pt.onPage):
                self.onPage(canv, doc)
                oop(canv, doc)
        else:

            def onPage(canv, doc):
                self.onPage(canv, doc)
        pt.onPage = onPage
        if pt.onPageEnd:

            def onPageEnd(canv, doc, oop=pt.onPageEnd):
                self.onPageEnd(canv, doc)
                oop(canv, doc)
        else:

            def onPageEnd(canv, doc):
                self.onPageEnd(canv, doc)
        pt.onPageEnd = onPageEnd

    def onPage(self, canv, doc):
        """this will be called at the start of the page"""
        setattr(canv, self.name, self)
        self.reset()

    def onPageEnd(self, canv, doc):
        """this will be called at the end of a page"""
        self.pageEndAction(canv, doc)
        try:
            delattr(canv, self.name)
        except:
            pass
        self.reset()

    def pageEndAction(self, canv, doc):
        """this should be overridden to do something useful"""
        pass

    def onDrawStr(self, value, *args):
        return onDrawStr(value, self, encode_label(args))