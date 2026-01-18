from reportlab.lib.units import inch
from reportlab.graphics.barcode.common import Barcode
from string import digits as string_digits, whitespace as string_whitespace
from reportlab.lib.utils import asNative
class POSTNET(Barcode):
    """
    POSTNET is used in the US to encode "zip codes" (postal codes) on
    mail. It can encode 5, 9, or 11 digit codes. I've read that it's
    pointless to do 5 digits, since USPS will just have to re-print
    them with 9 or 11 digits.

    Sources of information on POSTNET:

    USPS Publication 25, A Guide to Business Mail Preparation
    http://new.usps.com/cpim/ftp/pubs/pub25.pdf
    """
    quiet = 0
    shortHeight = inch * 0.05
    barHeight = inch * 0.125
    barWidth = inch * 0.018
    spaceWidth = inch * 0.0275

    def __init__(self, value='', **args):
        value = str(value) if isinstance(value, int) else asNative(value)
        for k, v in args.items():
            setattr(self, k, v)
        Barcode.__init__(self, value)

    def validate(self):
        self.validated = ''
        self.valid = 1
        count = 0
        for c in self.value:
            if c in string_whitespace + '-':
                pass
            elif c in string_digits:
                count = count + 1
                if count == 6:
                    self.validated = self.validated + '-'
                self.validated = self.validated + c
            else:
                self.valid = 0
        if len(self.validated) not in [5, 10, 12]:
            self.valid = 0
        return self.validated

    def encode(self):
        self.encoded = 'S'
        check = 0
        for c in self.validated:
            if c in string_digits:
                self.encoded = self.encoded + c
                check = check + int(c)
            elif c == '-':
                pass
            else:
                raise ValueError('Invalid character in input')
        check = (10 - check) % 10
        self.encoded = self.encoded + repr(check) + 'S'
        return self.encoded

    def decompose(self):
        self.decomposed = ''
        for c in self.encoded:
            self.decomposed = self.decomposed + _postnet_patterns[c]
        return self.decomposed

    def computeSize(self):
        self._width = len(self.decomposed) * self.barWidth + (len(self.decomposed) - 1) * self.spaceWidth
        self._height = self.barHeight

    def draw(self):
        self._calculate()
        sdown = self.barHeight - self.shortHeight
        left = 0
        for c in self.decomposed:
            if c == '.':
                h = self.shortHeight
            else:
                h = self.barHeight
            self.rect(left, 0.0, self.barWidth, h)
            left = left + self.barWidth + self.spaceWidth
        self.drawHumanReadable()

    def _humanText(self):
        return self.encoded[1:-1]