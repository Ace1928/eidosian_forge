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
def _calcSpanRects(self):
    """Work out rects for tables which do row and column spanning.

        Based on self._spanRanges, which is already known,
        and the widths which were given or previously calculated,
        self._spanRects shows the real coords for drawing:

            (col, row) -> (x, y, width, height)

        for each cell.  Any cell which 'does not exist' as another
        has spanned over it will get a None entry on the right
        """
    spanRects = getattr(self, '_spanRects', {})
    hmax = getattr(self, '_hmax', None)
    longTable = self._longTableOptimize
    if spanRects and (longTable and hmax == self._hmax_spanRects or not longTable):
        return
    colpositions = self._colpositions
    rowpositions = self._rowpositions
    vBlocks = {}
    hBlocks = {}
    rlim = len(rowpositions) - 1
    for coord, value in self._spanRanges.items():
        if value is None:
            spanRects[coord] = None
        else:
            try:
                col0, row0, col1, row1 = value
                if row1 >= rlim:
                    continue
                col, row = coord
                if col1 - col0 > 0:
                    for _ in range(col0 + 1, col1 + 1):
                        vBlocks.setdefault(colpositions[_], []).append((rowpositions[row1 + 1], rowpositions[row0]))
                if row1 - row0 > 0:
                    for _ in range(row0 + 1, row1 + 1):
                        hBlocks.setdefault(rowpositions[_], []).append((colpositions[col0], colpositions[col1 + 1]))
                x = colpositions[col0]
                y = rowpositions[row1 + 1]
                width = colpositions[col1 + 1] - x
                height = rowpositions[row0] - y
                spanRects[coord] = (x, y, width, height)
            except:
                annotateException('\nspanning problem in %s' % (self.identity(),))
    for _ in (hBlocks, vBlocks):
        for value in _.values():
            value.sort()
    self._spanRects = spanRects
    self._vBlocks = vBlocks
    self._hBlocks = hBlocks
    self._hmax_spanRects = hmax