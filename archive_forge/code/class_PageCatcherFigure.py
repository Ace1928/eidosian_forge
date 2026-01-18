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
class PageCatcherFigure(PageCatcherFigureNonA4):
    """PageCatcher page with a caption below it.  Presumes A4, Portrait.
        This needs our commercial PageCatcher product, or you'll get a blank."""

    def __init__(self, filename, pageNo, caption, width=595, height=842, background=None, caching=None):
        PageCatcherFigureNonA4.__init__(self, filename, pageNo, caption, width, height, background=background, caching=caching)