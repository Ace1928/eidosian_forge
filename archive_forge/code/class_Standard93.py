from reportlab.lib.units import inch
from reportlab.lib.utils import asNative
from reportlab.graphics.barcode.common import MultiWidthBarcode
class Standard93(_Code93Base):
    """
    Code 93 is a Uppercase alphanumeric symbology with some punctuation.
    See Extended Code 93 for a variant that can represent the entire
    128 characrter ASCII set.
    
    Options that may be passed to constructor:

        value (int, or numeric string. required.):
            The value to encode.
   
        barWidth (float, default .0075):
            X-Dimension, or width of the smallest element
            Minumum is .0075 inch (7.5 mils).
            
        barHeight (float, see default below):
            Height of the symbol.  Default is the height of the two
            bearer bars (if they exist) plus the greater of .25 inch
            or .15 times the symbol's length.

        quiet (bool, default 1):
            Wether to include quiet zones in the symbol.
            
        lquiet (float, see default below):
            Quiet zone size to left of code, if quiet is true.
            Default is the greater of .25 inch, or 10 barWidth
            
        rquiet (float, defaults as above):
            Quiet zone size to right left of code, if quiet is true.

        stop (bool, default 1):
            Whether to include start/stop symbols.

    Sources of Information on Code 93:

    http://www.semiconductor.agilent.com/barcode/sg/Misc/code_93.html

    Official Spec, "NSI/AIM BC5-1995, USS" is available for US$45 from
    http://www.aimglobal.org/aimstore/
    """

    def validate(self):
        vval = ''
        self.valid = 1
        for c in self.value.upper():
            if c not in _patterns:
                self.valid = 0
                continue
            vval = vval + c
        self.validated = vval
        return vval

    def encode(self):
        self.encoded = _encode93(self.validated)
        return self.encoded