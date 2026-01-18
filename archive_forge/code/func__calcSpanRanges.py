from reportlab.platypus.flowables import Flowable, Preformatted
from reportlab import rl_config
from reportlab.lib.styles import PropertySet, ParagraphStyle, _baseFontName
from reportlab.lib import colors
from reportlab.lib.utils import annotateException, IdentStr, flatten, isStr, asNative, strTypes, __UNSET__
from reportlab.lib.validators import isListOfNumbersOrNone
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.abag import ABag as CellFrame
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus.doctemplate import Indenter, NullActionFlowable
from reportlab.platypus.flowables import LIIndenter
from collections import namedtuple
def _calcSpanRanges(self):
    """Work out rects for tables which do row and column spanning.

        This creates some mappings to let the later code determine
        if a cell is part of a "spanned" range.
        self._spanRanges shows the 'coords' in integers of each
        'cell range', or None if it was clobbered:
        (col, row) -> (col0, row0, col1, row1)

        Any cell not in the key is not part of a spanned region
        """
    self._spanRanges = spanRanges = {}
    for x in range(self._ncols):
        for y in range(self._nrows):
            spanRanges[x, y] = (x, y, x, y)
    self._colSpanCells = []
    self._rowSpanCells = []
    csa = self._colSpanCells.append
    rsa = self._rowSpanCells.append
    for cmd, start, stop in self._spanCmds:
        x0, y0 = start
        x1, y1 = stop
        if x0 < 0:
            x0 = x0 + self._ncols
        if x1 < 0:
            x1 = x1 + self._ncols
        if y0 < 0:
            y0 = y0 + self._nrows
        if y1 < 0:
            y1 = y1 + self._nrows
        if x0 > x1:
            x0, x1 = (x1, x0)
        if y0 > y1:
            y0, y1 = (y1, y0)
        if x0 != x1 or y0 != y1:
            if x0 != x1:
                for y in range(y0, y1 + 1):
                    for x in range(x0, x1 + 1):
                        csa((x, y))
            if y0 != y1:
                for y in range(y0, y1 + 1):
                    for x in range(x0, x1 + 1):
                        rsa((x, y))
            for y in range(y0, y1 + 1):
                for x in range(x0, x1 + 1):
                    spanRanges[x, y] = None
            spanRanges[x0, y0] = (x0, y0, x1, y1)