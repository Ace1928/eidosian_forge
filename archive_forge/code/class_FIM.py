from reportlab.lib.units import inch
from reportlab.graphics.barcode.common import Barcode
from string import digits as string_digits, whitespace as string_whitespace
from reportlab.lib.utils import asNative
class FIM(Barcode):
    """
    FIM (Facing ID Marks) encode only one letter.
    There are currently four defined:

    A   Courtesy reply mail with pre-printed POSTNET
    B   Business reply mail without pre-printed POSTNET
    C   Business reply mail with pre-printed POSTNET
    D   OCR Readable mail without pre-printed POSTNET

    Options that may be passed to constructor:

        value (single character string from the set A - D. required.):
            The value to encode.

        quiet (bool, default 0):
            Whether to include quiet zones in the symbol.

    The following may also be passed, but doing so will generate nonstandard
    symbols which should not be used. This is mainly documented here to
    show the defaults:

        barHeight (float, default 5/8 inch):
            Height of the code. This might legitimately be overriden to make
            a taller symbol that will 'bleed' off the edge of the paper,
            leaving 5/8 inch remaining.

        lquiet (float, default 1/4 inch):
            Quiet zone size to left of code, if quiet is true.
            Default is the greater of .25 inch, or .15 times the symbol's
            length.

        rquiet (float, default 15/32 inch):
            Quiet zone size to right left of code, if quiet is true.

    Sources of information on FIM:

    USPS Publication 25, A Guide to Business Mail Preparation
    http://new.usps.com/cpim/ftp/pubs/pub25.pdf
    """
    barWidth = inch * (1.0 / 32.0)
    spaceWidth = inch * (1.0 / 16.0)
    barHeight = inch * (5.0 / 8.0)
    rquiet = inch * 0.25
    lquiet = inch * (15.0 / 32.0)
    quiet = 0

    def __init__(self, value='', **args):
        value = str(value) if isinstance(value, int) else asNative(value)
        for k, v in args.items():
            setattr(self, k, v)
        Barcode.__init__(self, value)

    def validate(self):
        self.valid = 1
        self.validated = ''
        for c in self.value:
            if c in string_whitespace:
                continue
            elif c in 'abcdABCD':
                self.validated = self.validated + c.upper()
            else:
                self.valid = 0
        if len(self.validated) != 1:
            raise ValueError('Input must be exactly one character')
        return self.validated

    def decompose(self):
        self.decomposed = ''
        for c in self.encoded:
            self.decomposed = self.decomposed + _fim_patterns[c]
        return self.decomposed

    def computeSize(self):
        self._width = (len(self.decomposed) - 1) * self.spaceWidth + self.barWidth
        if self.quiet:
            self._width += self.lquiet + self.rquiet
        self._height = self.barHeight

    def draw(self):
        self._calculate()
        left = self.quiet and self.lquiet or 0
        for c in self.decomposed:
            if c == '|':
                self.rect(left, 0.0, self.barWidth, self.barHeight)
            left += self.spaceWidth
        self.drawHumanReadable()

    def _humanText(self):
        return self.value