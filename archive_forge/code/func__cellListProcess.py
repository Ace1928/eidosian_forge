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
def _cellListProcess(self, v, aW, aH):
    if isinstance(v, _ExpandedCellTuple):
        C = v
    else:
        C = (v,) if isinstance(v, Flowable) else flatten(v)
        frame = None
        R = [].append
        for c in self._cellListIter(C, aW, aH):
            if isinstance(c, Indenter):
                if not frame:
                    frame = CellFrame(_leftExtraIndent=0, _rightExtraIndent=0)
                c.frameAction(frame)
                if frame._leftExtraIndent < 1e-08 and frame._rightExtraIndent < 1e-08:
                    frame = None
                continue
            if frame:
                R(LIIndenter(c, leftIndent=frame._leftExtraIndent, rightIndent=frame._rightExtraIndent))
            else:
                R(c)
        if hasattr(v, 'tagType'):
            C = _ExpandedCellTupleEx(R.__self__, v.tagType, v.altText, v.extras)
        else:
            C = _ExpandedCellTuple(R.__self__)
    return C