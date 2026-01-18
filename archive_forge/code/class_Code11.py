from reportlab.platypus.flowables import Flowable
from reportlab.lib.units import inch
from string import ascii_lowercase, ascii_uppercase, digits as string_digits
class Code11(Barcode):
    """
    Code 11 is an almost-numeric barcode. It encodes the digits 0-9 plus
    dash ("-"). 11 characters total, hence the name.

        value (int or string required.):
            The value to encode.

        barWidth (float, default .0075):
            X-Dimension, or width of the smallest element

        ratio (float, default 2.2):
            The ratio of wide elements to narrow elements.

        gap (float or None, default None):
            width of intercharacter gap. None means "use barWidth".

        barHeight (float, see default below):
            Height of the symbol.  Default is the height of the two
            bearer bars (if they exist) plus the greater of .25 inch
            or .15 times the symbol's length.

        checksum (0 none, 1 1-digit, 2 2-digit, -1 auto, default -1):
            How many checksum digits to include. -1 ("auto") means
            1 if the number of digits is 10 or less, else 2.

        bearers (float, in units of barWidth. default 0):
            Height of bearer bars (horizontal bars along the top and
            bottom of the barcode). Default is 0 (no bearers).

        quiet (bool, default 1):
            Wether to include quiet zones in the symbol.

        lquiet (float, see default below):
            Quiet zone size to left of code, if quiet is true.
            Default is the greater of .25 inch, or 10 barWidth

        rquiet (float, defaults as above):
            Quiet zone size to right left of code, if quiet is true.

    Sources of Information on Code 11:

    http://www.cwi.nl/people/dik/english/codes/barcodes.html
    """
    chars = '0123456789-'
    patterns = {'0': 'bsbsB', '1': 'BsbsB', '2': 'bSbsB', '3': 'BSbsb', '4': 'bsBsB', '5': 'BsBsb', '6': 'bSBsb', '7': 'bsbSB', '8': 'BsbSb', '9': 'Bsbsb', '-': 'bsBsb', 'S': 'bsBSb'}
    values = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '-': 10}
    stop = 1
    barHeight = None
    barWidth = inch * 0.0075
    ratio = 2.2
    checksum = -1
    bearers = 0.0
    quiet = 1
    lquiet = None
    rquiet = None

    def __init__(self, value='', **args):
        if type(value) == type(1):
            value = str(value)
        for k, v in args.items():
            setattr(self, k, v)
        if self.quiet:
            if self.lquiet is None:
                self.lquiet = min(inch * 0.25, self.barWidth * 10.0)
                self.rquiet = min(inch * 0.25, self.barWidth * 10.0)
        else:
            self.lquiet = self.rquiet = 0.0
        Barcode.__init__(self, value)

    def validate(self):
        vval = ''
        self.valid = 1
        s = self.value.strip()
        for i in range(0, len(s)):
            c = s[i]
            if c not in self.chars:
                self.Valid = 0
                continue
            vval = vval + c
        self.validated = vval
        return vval

    def _addCSD(self, s, m):
        i = c = 0
        v = 1
        V = self.values
        while i < len(s):
            c += v * V[s[-(i + 1)]]
            i += 1
            v += 1
            if v == m:
                v = 1
        return s + self.chars[c % 11]

    def encode(self):
        s = self.validated
        tcs = self.checksum
        if tcs < 0:
            self.checksum = tcs = 1 + int(len(s) > 10)
        if tcs > 0:
            s = self._addCSD(s, 11)
        if tcs > 1:
            s = self._addCSD(s, 10)
        self.encoded = self.stop and 'S' + s + 'S' or s

    def decompose(self):
        self.decomposed = ''.join([self.patterns[c] + 'i' for c in self.encoded])[:-1]
        return self.decomposed

    def _humanText(self):
        return self.stop and self.encoded[1:-1] or self.encoded