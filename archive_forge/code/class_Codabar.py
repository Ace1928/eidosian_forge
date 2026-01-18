from reportlab.platypus.flowables import Flowable
from reportlab.lib.units import inch
from string import ascii_lowercase, ascii_uppercase, digits as string_digits
class Codabar(Barcode):
    """
    Codabar is a numeric plus some puntuation ("-$:/.+") barcode
    with four start/stop characters (A, B, C, and D).

    Options that may be passed to constructor:

        value (string required.):
            The value to encode.

        barWidth (float, default .0065):
            X-Dimension, or width of the smallest element
            minimum is 6.5 mils (.0065 inch)

        ratio (float, default 2.0):
            The ratio of wide elements to narrow elements.

        gap (float or None, default None):
            width of intercharacter gap. None means "use barWidth".

        barHeight (float, see default below):
            Height of the symbol.  Default is the height of the two
            bearer bars (if they exist) plus the greater of .25 inch
            or .15 times the symbol's length.

        checksum (bool, default 0):
            Whether to compute and include the check digit

        bearers (float, in units of barWidth. default 0):
            Height of bearer bars (horizontal bars along the top and
            bottom of the barcode). Default is 0 (no bearers).

        quiet (bool, default 1):
            Whether to include quiet zones in the symbol.

        stop (bool, default 1):
            Whether to include start/stop symbols.

        lquiet (float, see default below):
            Quiet zone size to left of code, if quiet is true.
            Default is the greater of .25 inch, or 10 barWidth

        rquiet (float, defaults as above):
            Quiet zone size to right left of code, if quiet is true.

    Sources of Information on Codabar

    http://www.semiconductor.agilent.com/barcode/sg/Misc/codabar.html
    http://www.barcodeman.com/codabar.html

    Official Spec, "ANSI/AIM BC3-1995, USS" is available for US$45 from
    http://www.aimglobal.org/aimstore/
    """
    patterns = {'0': 'bsbsbSB', '1': 'bsbsBSb', '2': 'bsbSbsB', '3': 'BSbsbsb', '4': 'bsBsbSb', '5': 'BsbsbSb', '6': 'bSbsbsB', '7': 'bSbsBsb', '8': 'bSBsbsb', '9': 'BsbSbsb', '-': 'bsbSBsb', '$': 'bsBSbsb', ':': 'BsbsBsB', '/': 'BsBsbsB', '.': 'BsBsBsb', '+': 'bsBsBsB', 'A': 'bsBSbSb', 'B': 'bSbSbsB', 'C': 'bsbSbSB', 'D': 'bsbSBSb'}
    values = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '-': 10, '$': 11, ':': 12, '/': 13, '.': 14, '+': 15, 'A': 16, 'B': 17, 'C': 18, 'D': 19}
    chars = string_digits + '-$:/.+'
    stop = 1
    barHeight = None
    barWidth = inch * 0.0065
    ratio = 2.0
    checksum = 0
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
                if i != 0 and i != len(s) - 1 or c not in 'ABCD':
                    self.Valid = 0
                    continue
            vval = vval + c
        if self.stop:
            if vval[0] not in 'ABCD':
                vval = 'A' + vval
            if vval[-1] not in 'ABCD':
                vval = vval + vval[0]
        self.validated = vval
        return vval

    def encode(self):
        s = self.validated
        if self.checksum:
            v = sum([self.values[c] for c in s])
            s += self.chars[v % 16]
        self.encoded = s

    def decompose(self):
        dval = ''.join([self.patterns[c] + 'i' for c in self.encoded])
        self.decomposed = dval[:-1]
        return self.decomposed