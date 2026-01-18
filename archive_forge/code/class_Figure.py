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
class Figure(Flowable):

    def __init__(self, width, height, caption='', captionFont=_baseFontNameI, captionSize=12, background=None, captionTextColor=toColor('black'), captionBackColor=None, border=None, spaceBefore=12, spaceAfter=12, captionGap=None, captionAlign='centre', captionPosition='bottom', hAlign='CENTER'):
        Flowable.__init__(self)
        self.width = width
        self.figureHeight = height
        self.caption = caption
        self.captionFont = captionFont
        self.captionSize = captionSize
        self.captionTextColor = captionTextColor
        self.captionBackColor = captionBackColor
        self.captionGap = captionGap or 0.5 * captionSize
        self.captionAlign = captionAlign
        self.captionPosition = captionPosition
        self._captionData = None
        self.captionHeight = 0
        self.background = background
        self.border = border
        self.spaceBefore = spaceBefore
        self.spaceAfter = spaceAfter
        self.hAlign = hAlign
        self._getCaptionPara()

    def _getCaptionPara(self):
        caption = self.caption
        captionFont = self.captionFont
        captionSize = self.captionSize
        captionTextColor = self.captionTextColor
        captionBackColor = self.captionBackColor
        captionAlign = self.captionAlign
        captionPosition = self.captionPosition
        if self._captionData != (caption, captionFont, captionSize, captionTextColor, captionBackColor, captionAlign, captionPosition):
            self._captionData = (caption, captionFont, captionSize, captionTextColor, captionBackColor, captionAlign, captionPosition)
            if isinstance(caption, Paragraph):
                self.captionPara = caption
            elif isinstance(caption, strTypes):
                self.captionStyle = ParagraphStyle('Caption', fontName=captionFont, fontSize=captionSize, leading=1.2 * captionSize, textColor=captionTextColor, backColor=captionBackColor, spaceBefore=self.captionGap, alignment=TA_LEFT if captionAlign == 'left' else TA_RIGHT if captionAlign == 'right' else TA_CENTER)
                self.captionPara = Paragraph(self.caption, self.captionStyle)
            else:
                raise ValueError('Figure caption of type %r is not a string or Paragraph' % type(caption))

    def wrap(self, availWidth, availHeight):
        if self.caption:
            self._getCaptionPara()
            w, h = self.captionPara.wrap(self.width, availHeight - self.figureHeight)
            self.captionHeight = h + self.captionGap
            self.height = self.captionHeight + self.figureHeight
            if w > self.width:
                self.width = w
        else:
            self.height = self.figureHeight
        if self.hAlign in ('CENTER', 'CENTRE', TA_CENTER):
            self.dx = 0.5 * (availWidth - self.width)
        elif self.hAlign in ('RIGHT', TA_RIGHT):
            self.dx = availWidth - self.width
        else:
            self.dx = 0
        return (self.width, self.height)

    def draw(self):
        self.canv.translate(self.dx, 0)
        if self.caption and self.captionPosition == 'bottom':
            self.canv.translate(0, self.captionHeight)
        if self.background:
            self.drawBackground()
        if self.border:
            self.drawBorder()
        self.canv.saveState()
        self.drawFigure()
        self.canv.restoreState()
        if self.caption:
            if self.captionPosition == 'bottom':
                self.canv.translate(0, -self.captionHeight)
            else:
                self.canv.translate(0, self.figureHeight + self.captionGap)
            self._getCaptionPara()
            self.drawCaption()

    def drawBorder(self):
        self.canv.drawBoundary(self.border, 0, 0, self.width, self.figureHeight)

    def _doBackground(self, color):
        self.canv.saveState()
        self.canv.setFillColor(self.background)
        self.canv.rect(0, 0, self.width, self.figureHeight, fill=1)
        self.canv.restoreState()

    def drawBackground(self):
        """For use when using a figure on a differently coloured background.
        Allows you to specify a colour to be used as a background for the figure."""
        if isColor(self.background):
            self._doBackground(self.background)
        else:
            try:
                c = toColor(self.background)
                self._doBackground(c)
            except:
                pass

    def drawCaption(self):
        self.captionPara.drawOn(self.canv, 0, 0)

    def drawFigure(self):
        pass