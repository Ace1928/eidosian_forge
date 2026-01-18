from reportlab.platypus.flowables import Flowable
from reportlab.lib.units import inch
from string import ascii_lowercase, ascii_uppercase, digits as string_digits
class MultiWidthBarcode(Barcode):
    """Base for variable-bar-width codes like Code93 and Code128"""

    def computeSize(self, *args):
        barWidth = self.barWidth
        oa, oA = (ord('a') - 1, ord('A') - 1)
        w = 0.0
        for c in self.decomposed:
            oc = ord(c)
            if c in ascii_lowercase:
                w = w + barWidth * (oc - oa)
            elif c in ascii_uppercase:
                w = w + barWidth * (oc - oA)
        if self.barHeight is None:
            self.barHeight = w * 0.15
            self.barHeight = max(0.25 * inch, self.barHeight)
        if self.quiet:
            w += self.lquiet + self.rquiet
        self._height = self.barHeight
        self._width = w

    def draw(self):
        self._calculate()
        oa, oA = (ord('a') - 1, ord('A') - 1)
        barWidth = self.barWidth
        left = self.quiet and self.lquiet or 0
        for c in self.decomposed:
            oc = ord(c)
            if c in ascii_lowercase:
                left = left + (oc - oa) * barWidth
            elif c in ascii_uppercase:
                w = (oc - oA) * barWidth
                self.rect(left, 0, w, self.barHeight)
                left += w
        self.drawHumanReadable()