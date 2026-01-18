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
class FlexFigure(Figure):
    """Base for a figure class with a caption. Can grow or shrink in proportion"""

    def __init__(self, width, height, caption, background=None, captionFont='Helvetica-Oblique', captionSize=8, captionTextColor=colors.black, shrinkToFit=1, growToFit=1, spaceBefore=12, spaceAfter=12, captionGap=9, captionAlign='centre', captionPosition='top', scaleFactor=None, hAlign='CENTER', border=1):
        Figure.__init__(self, width, height, caption, captionFont=captionFont, captionSize=captionSize, background=None, captionTextColor=captionTextColor, spaceBefore=spaceBefore, spaceAfter=spaceAfter, captionGap=captionGap, captionAlign=captionAlign, captionPosition=captionPosition, hAlign=hAlign, border=border)
        self.shrinkToFit = shrinkToFit
        self.growToFit = growToFit
        self.scaleFactor = scaleFactor
        self._scaleFactor = None
        self.background = background

    def _scale(self, availWidth, availHeight):
        """Rescale to fit according to the rules, but only once"""
        if self._scaleFactor is None or self.width > availWidth or self.height > availHeight:
            w, h = Figure.wrap(self, availWidth, availHeight)
            captionHeight = h - self.figureHeight
            if self.scaleFactor is None:
                self._scaleFactor = min(availWidth / self.width, (availHeight - captionHeight) / self.figureHeight)
            else:
                self._scaleFactor = self.scaleFactor
            if self._scaleFactor < 1 and self.shrinkToFit:
                self.width = self.width * self._scaleFactor - 0.0001
                self.figureHeight = self.figureHeight * self._scaleFactor
            elif self._scaleFactor > 1 and self.growToFit:
                self.width = self.width * self._scaleFactor - 0.0001
                self.figureHeight = self.figureHeight * self._scaleFactor

    def wrap(self, availWidth, availHeight):
        self._scale(availWidth, availHeight)
        return Figure.wrap(self, availWidth, availHeight)

    def split(self, availWidth, availHeight):
        self._scale(availWidth, availHeight)
        return Figure.split(self, availWidth, availHeight)