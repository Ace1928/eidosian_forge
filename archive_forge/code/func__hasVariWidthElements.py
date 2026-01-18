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
def _hasVariWidthElements(self, upToRow=None):
    """Check for flowables in table cells and warn up front.

        Allow a couple which we know are fixed size such as
        images and graphics."""
    if upToRow is None:
        upToRow = self._nrows
    for row in range(min(self._nrows, upToRow)):
        for col in range(self._ncols):
            value = self._cellvalues[row][col]
            if not self._canGetWidth(value):
                return 1
    return 0