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
class PageCatcherCachingMixIn:
    """Helper functions to cache pages for figures"""

    def getFormName(self, pdfFileName, pageNo):
        dirname, filename = os.path.split(pdfFileName)
        root, ext = os.path.splitext(filename)
        return '%s_page%d' % (root, pageNo)

    def needsProcessing(self, pdfFileName, pageNo):
        """returns 1 if no forms or form is older"""
        formName = self.getFormName(pdfFileName, pageNo)
        if os.path.exists(formName + '.frm'):
            formModTime = os.stat(formName + '.frm')[8]
            pdfModTime = os.stat(pdfFileName)[8]
            return pdfModTime > formModTime
        else:
            return 1

    def processPDF(self, pdfFileName, pageNo):
        formName = self.getFormName(pdfFileName, pageNo)
        storeForms(pdfFileName, formName + '.frm', prefix=formName + '_', pagenumbers=[pageNo])
        return formName + '.frm'