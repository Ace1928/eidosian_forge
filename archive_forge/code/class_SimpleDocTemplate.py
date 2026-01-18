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
class SimpleDocTemplate(BaseDocTemplate):
    """A special case document template that will handle many simple documents.
       See documentation for BaseDocTemplate.  No pageTemplates are required
       for this special case.   A page templates are inferred from the
       margin information and the onFirstPage, onLaterPages arguments to the build method.

       A document which has all pages with the same look except for the first
       page may can be built using this special approach.
    """
    _invalidInitArgs = ('pageTemplates',)

    def handle_pageBegin(self):
        """override base method to add a change of page template after the firstpage.
        """
        self._handle_pageBegin()
        self._handle_nextPageTemplate('Later')

    def build(self, flowables, onFirstPage=_doNothing, onLaterPages=_doNothing, canvasmaker=canvas.Canvas):
        """build the document using the flowables.  Annotate the first page using the onFirstPage
               function and later pages using the onLaterPages function.  The onXXX pages should follow
               the signature

                  def myOnFirstPage(canvas, document):
                      # do annotations and modify the document
                      ...

               The functions can do things like draw logos, page numbers,
               footers, etcetera. They can use external variables to vary
               the look (for example providing page numbering or section names).
        """
        self._calc()
        frameT = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id='normal')
        self.addPageTemplates([PageTemplate(id='First', frames=frameT, onPage=onFirstPage, pagesize=self.pagesize), PageTemplate(id='Later', frames=frameT, onPage=onLaterPages, pagesize=self.pagesize)])
        if onFirstPage is _doNothing and hasattr(self, 'onFirstPage'):
            self.pageTemplates[0].beforeDrawPage = self.onFirstPage
        if onLaterPages is _doNothing and hasattr(self, 'onLaterPages'):
            self.pageTemplates[1].beforeDrawPage = self.onLaterPages
        BaseDocTemplate.build(self, flowables, canvasmaker=canvasmaker)