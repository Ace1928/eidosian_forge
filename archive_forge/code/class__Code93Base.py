from reportlab.lib.units import inch
from reportlab.lib.utils import asNative
from reportlab.graphics.barcode.common import MultiWidthBarcode
class _Code93Base(MultiWidthBarcode):
    barWidth = inch * 0.0075
    lquiet = None
    rquiet = None
    quiet = 1
    barHeight = None
    stop = 1

    def __init__(self, value='', **args):
        if type(value) is type(1):
            value = asNative(value)
        for k, v in args.items():
            setattr(self, k, v)
        if self.quiet:
            if self.lquiet is None:
                self.lquiet = max(inch * 0.25, self.barWidth * 10.0)
                self.rquiet = max(inch * 0.25, self.barWidth * 10.0)
        else:
            self.lquiet = self.rquiet = 0.0
        MultiWidthBarcode.__init__(self, value)

    def decompose(self):
        dval = self.stop and [_patterns['start'][0]] or []
        dval += [_patterns[c][0] for c in self.encoded]
        if self.stop:
            dval.append(_patterns['stop'][0])
        self.decomposed = ''.join(dval)
        return self.decomposed