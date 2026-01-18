from reportlab.graphics.barcode.code39 import Standard39
from reportlab.lib import colors
from reportlab.lib.units import cm
from string import ascii_uppercase, digits as string_digits
class BaseLTOLabel(Standard39):
    """
    Base class for LTO labels.

    Specification taken from "IBM LTO Ultrium Cartridge Label Specification, Revision 3"
    available on  May 14th 2008 from :
    http://www-1.ibm.com/support/docview.wss?rs=543&context=STCVQ6R&q1=ssg1*&uid=ssg1S7000429&loc=en_US&cs=utf-8&lang=en+en
    """
    LABELWIDTH = 7.9 * cm
    LABELHEIGHT = 1.7 * cm
    LABELROUND = 0.15 * cm
    CODERATIO = 2.75
    CODENOMINALWIDTH = 7.4088 * cm
    CODEBARHEIGHT = 1.11 * cm
    CODEBARWIDTH = 0.0432 * cm
    CODEGAP = CODEBARWIDTH
    CODELQUIET = 10 * CODEBARWIDTH
    CODERQUIET = 10 * CODEBARWIDTH

    def __init__(self, prefix='', number=None, subtype='1', border=None, checksum=False, availheight=None):
        """
           Initializes an LTO label.

           prefix : Up to six characters from [A-Z][0-9]. Defaults to "".
           number : Label's number or None. Defaults to None.
           subtype : LTO subtype string , e.g. "1" for LTO1. Defaults to "1".
           border : None, or the width of the label's border. Defaults to None.
           checksum : Boolean indicates if checksum char has to be printed. Defaults to False.
           availheight : Available height on the label, or None for automatic. Defaults to None.
        """
        self.height = max(availheight, self.CODEBARHEIGHT)
        self.border = border
        if len(subtype) != 1 or subtype not in ascii_uppercase + string_digits:
            raise ValueError("Invalid subtype '%s'" % subtype)
        if not number and len(prefix) > 6 or not prefix.isalnum():
            raise ValueError("Invalid prefix '%s'" % prefix)
        label = '%sL%s' % ((prefix + str(number or 0).zfill(6 - len(prefix)))[:6], subtype)
        if len(label) != 8:
            raise ValueError('Invalid set of parameters (%s, %s, %s)' % (prefix, number, subtype))
        self.label = label
        Standard39.__init__(self, label, ratio=self.CODERATIO, barHeight=self.height, barWidth=self.CODEBARWIDTH, gap=self.CODEGAP, lquiet=self.CODELQUIET, rquiet=self.CODERQUIET, quiet=True, checksum=checksum)

    def drawOn(self, canvas, x, y):
        """Draws the LTO label onto the canvas."""
        canvas.saveState()
        canvas.translate(x, y)
        if self.border:
            canvas.setLineWidth(self.border)
            canvas.roundRect(0, 0, self.LABELWIDTH, self.LABELHEIGHT, self.LABELROUND)
        Standard39.drawOn(self, canvas, (self.LABELWIDTH - self.CODENOMINALWIDTH) / 2.0, self.LABELHEIGHT - self.height)
        canvas.restoreState()