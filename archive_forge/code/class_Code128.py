from reportlab.lib.units import inch
from reportlab.lib.utils import asNative
from reportlab.graphics.barcode.common import MultiWidthBarcode
from string import digits
class Code128(MultiWidthBarcode):
    """
    Code 128 is a very compact symbology that can encode the entire
    128 character ASCII set, plus 4 special control codes,
    (FNC1-FNC4, expressed in the input string as ñ to ô).
    Code 128 can also encode digits at double density (2 per byte)
    and has a mandatory checksum.  Code 128 is well supported and
    commonly used -- for example, by UPS for tracking labels.
    
    Because of these qualities, Code 128 is probably the best choice
    for a linear symbology today (assuming you have a choice).

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
            
    Sources of Information on Code 128:

    http://www.semiconductor.agilent.com/barcode/sg/Misc/code_128.html
    http://www.adams1.com/pub/russadam/128code.html
    http://www.barcodeman.com/c128.html

    Official Spec, "ANSI/AIM BC4-1999, ISS" is available for US$45 from
    http://www.aimglobal.org/aimstore/
    """
    barWidth = inch * 0.0075
    lquiet = None
    rquiet = None
    quiet = 1
    barHeight = None

    def __init__(self, value='', **args):
        value = str(value) if isinstance(value, int) else asNative(value)
        for k, v in args.items():
            setattr(self, k, v)
        if self.quiet:
            if self.lquiet is None:
                self.lquiet = max(inch * 0.25, self.barWidth * 10.0)
            if self.rquiet is None:
                self.rquiet = max(inch * 0.25, self.barWidth * 10.0)
        else:
            self.lquiet = self.rquiet = 0.0
        MultiWidthBarcode.__init__(self, value)

    def validate(self):
        vval = ''
        self.valid = 1
        for c in self.value:
            if ord(c) > 127 and c not in 'ñòóô':
                self.valid = 0
                continue
            vval = vval + c
        self.validated = vval
        return vval

    def _try_TO_C(self, l):
        """Improved version of old _trailingDigitsToC(self, l) inspired by"""
        i = 0
        nl = []
        while i < len(l):
            startpos = i
            rl = []
            savings = -1
            while i < len(l):
                if l[i] in cStarts:
                    j = i
                    break
                elif l[i] == 'ñ':
                    rl.append(l[i])
                    i += 1
                    continue
                elif l[i] in digits and l[i + 1] in digits:
                    rl.append(l[i] + l[i + 1])
                    i += 2
                    savings += 1
                    continue
                else:
                    if l[i] in digits and l[i + 1] == 'STOP':
                        rrl = []
                        rsavings = -1
                        k = i
                        while k > startpos:
                            if l[k] == 'ñ':
                                rrl.append(l[i])
                                k -= 1
                            elif l[k] in digits and l[k - 1] in digits:
                                rrl.append(l[k - 1] + l[k])
                                rsavings += 1
                                k -= 2
                            else:
                                break
                        rrl.reverse()
                        if rsavings > savings + int(savings >= 0 and (startpos and nl[-1] in cStarts)) - 1:
                            nl += l[startpos]
                            startpos += 1
                            rl = rrl
                            del rrl
                            i += 1
                    break
            ta = not (l[i] == 'STOP' or j == i)
            xs = savings >= 0 and (startpos and nl[-1] in cStarts)
            if savings + int(xs) > int(ta):
                if xs:
                    toc = nl[-1][:-1] + 'C'
                    del nl[-1]
                else:
                    toc = 'TO_C'
                nl += [toc] + rl
                if ta:
                    nl.append('TO' + l[j][-2:])
                nl.append(l[i])
            else:
                nl += l[startpos:i + 1]
            i += 1
        return nl

    def encode(self):
        s = self.validated
        l = ['START_B']
        for c in s:
            if c not in setb:
                l = l + ['TO_A', c, 'TO_B']
            else:
                l.append(c)
        l.append('STOP')
        l = self._try_TO_C(l)
        if l[1] in tos:
            l[:2] = ['START_' + l[1][-1]]
        start, set, shset = setmap[l[0]]
        e = [start]
        l = l[1:-1]
        while l:
            c = l[0]
            if c == 'SHIFT':
                e = e + [set[c], shset[l[1]]]
                l = l[2:]
            elif c in tos:
                e.append(set[c])
                set, shset = setmap[c]
                l = l[1:]
            else:
                e.append(set[c])
                l = l[1:]
        c = e[0]
        for i in range(1, len(e)):
            c = c + i * e[i]
        self.encoded = e + [c % 103, stop]
        return self.encoded

    def decompose(self):
        self.decomposed = ''.join([_patterns[c] for c in self.encoded])
        return self.decomposed

    def _humanText(self):
        return self.value