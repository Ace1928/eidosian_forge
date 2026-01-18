import os
from copy import deepcopy, copy
from reportlab.lib.colors import gray, lightgrey
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.lib.styles import _baseFontName
from reportlab.lib.utils import strTypes, rl_safe_exec, annotateException
from reportlab.lib.abag import ABag
from reportlab.pdfbase import pdfutils
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.rl_config import _FUZZ, overlapAttachedSpace, ignoreContainerActions, listWrapOnFakeWidth
from reportlab.lib.sequencer import _type2formatter
from reportlab.lib.styles import ListStyle
class Preformatted(Flowable):
    """This is like the HTML <PRE> tag.
    It attempts to display text exactly as you typed it in a fixed width "typewriter" font.
    By default the line breaks are exactly where you put them, and it will not be wrapped.
    You can optionally define a maximum line length and the code will be wrapped; and
    extra characters to be inserted at the beginning of each wrapped line (e.g. '> ').
    """

    def __init__(self, text, style, bulletText=None, dedent=0, maxLineLength=None, splitChars=None, newLineChars=''):
        """text is the text to display. If dedent is set then common leading space
        will be chopped off the front (for example if the entire text is indented
        6 spaces or more then each line will have 6 spaces removed from the front).
        """
        self.style = style
        self.bulletText = bulletText
        self.lines = _dedenter(text, dedent)
        if text and maxLineLength:
            self.lines = splitLines(self.lines, maxLineLength, splitChars, newLineChars)

    def __repr__(self):
        bT = self.bulletText
        H = 'Preformatted('
        if bT is not None:
            H = 'Preformatted(bulletText=%s,' % repr(bT)
        return "%s'''\\ \n%s''')" % (H, '\n'.join(self.lines))

    def wrap(self, availWidth, availHeight):
        self.width = availWidth
        self.height = self.style.leading * len(self.lines)
        return (self.width, self.height)

    def minWidth(self):
        style = self.style
        fontSize = style.fontSize
        fontName = style.fontName
        return max([stringWidth(line, fontName, fontSize) for line in self.lines])

    def split(self, availWidth, availHeight):
        if availHeight < self.style.leading:
            return []
        linesThatFit = int(availHeight * 1.0 / self.style.leading)
        text1 = '\n'.join(self.lines[0:linesThatFit])
        text2 = '\n'.join(self.lines[linesThatFit:])
        style = self.style
        if style.firstLineIndent != 0:
            style = deepcopy(style)
            style.firstLineIndent = 0
        return [Preformatted(text1, self.style), Preformatted(text2, style)]

    def draw(self):
        cur_x = self.style.leftIndent
        cur_y = self.height - self.style.fontSize
        self.canv.addLiteral('%PreformattedPara')
        if self.style.textColor:
            self.canv.setFillColor(self.style.textColor)
        tx = self.canv.beginText(cur_x, cur_y)
        tx.setFont(self.style.fontName, self.style.fontSize, self.style.leading)
        for text in self.lines:
            tx.textLine(text)
        self.canv.drawText(tx)