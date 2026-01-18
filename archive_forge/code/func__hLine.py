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
def _hLine(canvLine, scp, ecp, y, hBlocks, FUZZ=rl_config._FUZZ):
    """
    Draw horizontal lines; do not draw through regions specified in hBlocks
    This also serves for vertical lines with a suitable canvLine
    """
    if hBlocks:
        hBlocks = hBlocks.get(y, None)
    if not hBlocks or scp >= hBlocks[-1][1] - FUZZ or ecp <= hBlocks[0][0] + FUZZ:
        canvLine(scp, y, ecp, y)
    else:
        i = 0
        n = len(hBlocks)
        while scp < ecp - FUZZ and i < n:
            x0, x1 = hBlocks[i]
            if x1 <= scp + FUZZ or x0 >= ecp - FUZZ:
                i += 1
                continue
            i0 = max(scp, x0)
            i1 = min(ecp, x1)
            if i0 > scp:
                canvLine(scp, y, i0, y)
            scp = i1
        if scp < ecp - FUZZ:
            canvLine(scp, y, ecp, y)