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
def _drawHLines(self, start, end, weight, color, count, space):
    sc, sr = start
    ec, er = end
    ecp = self._colpositions[sc:ec + 2]
    rp = self._rowpositions[sr:er + 1]
    if len(ecp) <= 1 or len(rp) < 1:
        return
    self._prepLine(weight, color)
    scp = ecp[0]
    ecp = ecp[-1]
    hBlocks = getattr(self, '_hBlocks', {})
    canvLine = self.canv.line
    if count == 1:
        for y in rp:
            _hLine(canvLine, scp, ecp, y, hBlocks)
    else:
        lf = lambda x0, y0, x1, y1, canvLine=canvLine, ws=weight + space, count=count: _multiLine(x0, x1, y0, canvLine, ws, count)
        for y in rp:
            _hLine(lf, scp, ecp, y, hBlocks)