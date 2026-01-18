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
def _drawVLines(self, start, end, weight, color, count, space):
    sc, sr = start
    ec, er = end
    erp = self._rowpositions[sr:er + 2]
    cp = self._colpositions[sc:ec + 1]
    if len(erp) <= 1 or len(cp) < 1:
        return
    self._prepLine(weight, color)
    srp = erp[0]
    erp = erp[-1]
    vBlocks = getattr(self, '_vBlocks', {})
    canvLine = lambda y0, x0, y1, x1, _line=self.canv.line: _line(x0, y0, x1, y1)
    if count == 1:
        for x in cp:
            _hLine(canvLine, erp, srp, x, vBlocks)
    else:
        lf = lambda x0, y0, x1, y1, canvLine=canvLine, ws=weight + space, count=count: _multiLine(x0, x1, y0, canvLine, ws, count)
        for x in cp:
            _hLine(lf, erp, srp, x, vBlocks)