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
def _setCornerRadii(self, cornerRadii):
    if isListOfNumbersOrNone(cornerRadii):
        self._cornerRadii = None if not cornerRadii else list(cornerRadii) + max(4 - len(cornerRadii), 0) * [0]
    else:
        raise ValueError(f'cornerRadii should be None or a list/tuple of numeric radii\nnot {cornerRadii!a}')