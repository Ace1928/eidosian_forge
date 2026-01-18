import os
from reportlab.lib import colors
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import recursiveImport, strTypes
from reportlab.platypus import Frame
from reportlab.platypus import Flowable
from reportlab.platypus import Paragraph
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER
from reportlab.lib.validators import isColor
from reportlab.lib.colors import toColor
from reportlab.lib.styles import _baseFontName, _baseFontNameI
class cachePageCatcherFigureNonA4(FlexFigure, PageCatcherCachingMixIn):
    """PageCatcher page with a caption below it.  Size to be supplied."""

    def __init__(self, filename, pageNo, caption, width, height, background=None):
        self.dirname, self.filename = os.path.split(filename)
        if self.dirname == '':
            self.dirname = os.curdir
        self.pageNo = pageNo
        self.formName = self.getFormName(self.filename, self.pageNo) + '_' + str(pageNo)
        FlexFigure.__init__(self, width, height, caption, background)

    def drawFigure(self):
        self.canv.saveState()
        if not self.canv.hasForm(self.formName):
            restorePath = self.dirname + os.sep + self.filename
            formFileName = self.getFormName(restorePath, self.pageNo) + '.frm'
            if self.needsProcessing(restorePath, self.pageNo):
                self.processPDF(restorePath, self.pageNo)
            names = restoreForms(formFileName, self.canv)
        self.canv.scale(self._scaleFactor, self._scaleFactor)
        self.canv.doForm(self.formName)
        self.canv.restoreState()