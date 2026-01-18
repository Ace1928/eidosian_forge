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
def _makeCanvas(self, filename=None, canvasmaker=canvas.Canvas):
    """make and return a sample canvas. As suggested by 
        Chris Jerdonek cjerdonek @ bitbucket this allows testing of stringWidths
        etc.

        *NB* only the canvases created in self._startBuild will actually be used
        in the build process.
        """
    self.seq = reportlab.lib.sequencer.Sequencer()
    canv = canvasmaker(filename or self.filename, pagesize=self.pagesize, invariant=self.invariant, pageCompression=self.pageCompression, enforceColorSpace=self.enforceColorSpace, initialFontName=self.initialFontName, initialFontSize=self.initialFontSize, initialLeading=self.initialLeading, cropBox=self.cropBox, artBox=self.artBox, trimBox=self.trimBox, bleedBox=self.bleedBox, lang=self.lang)
    getattr(canv, 'setEncrypt', lambda x: None)(self.encrypt)
    canv._cropMarks = self.cropMarks
    canv.setAuthor(self.author)
    canv.setTitle(self.title)
    canv.setSubject(self.subject)
    canv.setCreator(self.creator)
    canv.setProducer(self.producer)
    canv.setKeywords(self.keywords)
    from reportlab.pdfbase.pdfdoc import ViewerPreferencesPDFDictionary as VPD, checkPDFBoolean as cPDFB
    for k, vf in VPD.validate.items():
        v = getattr(self, k[0].lower() + k[1:], None)
        if v is not None:
            if vf is cPDFB:
                v = ['false', 'true'][v]
            canv.setViewerPreference(k, v)
    if self._onPage:
        canv.setPageCallBack(self._onPage)
    return canv