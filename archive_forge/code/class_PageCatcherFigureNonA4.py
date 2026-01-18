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
class PageCatcherFigureNonA4(FlexFigure):
    """PageCatcher page with a caption below it.  Size to be supplied."""
    _cache = {}

    def __init__(self, filename, pageNo, caption, width, height, background=None, caching=None):
        fn = self.filename = filename
        self.pageNo = pageNo
        fn = fn.replace(os.sep, '_').replace('/', '_').replace('\\', '_').replace('-', '_').replace(':', '_')
        self.prefix = fn.replace('.', '_') + '_' + str(pageNo) + '_'
        self.formName = self.prefix + str(pageNo)
        self.caching = caching
        FlexFigure.__init__(self, width, height, caption, background)

    def drawFigure(self):
        if not self.canv.hasForm(self.formName):
            if self.filename in self._cache:
                f, data = self._cache[self.filename]
            else:
                f = open(self.filename, 'rb')
                pdf = f.read()
                f.close()
                f, data = storeFormsInMemory(pdf, pagenumbers=[self.pageNo], prefix=self.prefix)
                if self.caching == 'memory':
                    self._cache[self.filename] = (f, data)
            f = restoreFormsInMemory(data, self.canv)
        self.canv.saveState()
        self.canv.scale(self._scaleFactor, self._scaleFactor)
        self.canv.doForm(self.formName)
        self.canv.restoreState()