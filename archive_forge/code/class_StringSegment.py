import math
import xmllib
from rdkit.sping.pid import Font
from sping.PDF import PDFCanvas
class StringSegment:
    """class StringSegment contains the intermediate representation of string
        segments as they are being parsed by the XMLParser.
        """

    def __init__(self):
        self.super = 0
        self.sub = 0
        self.bold = 0
        self.italic = 0
        self.underline = 0
        self.s = ''
        self.width = 0
        self.greek = 0

    def calcNewFont(self, font):
        """Given a font (does not accept font==None), creates a                 new font based on the format of this text segment."""
        if self.greek:
            face = 'symbol'
        else:
            face = font.face
        return Font(face=face, size=font.size - self.super * sizedelta - self.sub * sizedelta, underline=self.underline or font.underline, bold=self.bold or font.bold, italic=self.italic or font.italic)

    def calcNewY(self, font, y):
        """Returns a new y coordinate depending on its                 whether the string is a sub and super script."""
        if self.sub == 1:
            return y + font.size * subFraction
        elif self.super == 1:
            return y - font.size * superFraction
        else:
            return y

    def dump(self):
        print('StringSegment: ]%s[' % self.s)
        print('\tsuper = ', self.super)
        print('\tsub = ', self.sub)
        print('\tbold = ', self.bold)
        print('\titalic = ', self.italic)
        print('\tunderline = ', self.underline)
        print('\twidth = ', self.width)
        print('\tgreek = ', self.greek)