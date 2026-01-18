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
class CellStyle(PropertySet):
    fontname = _baseFontName
    fontsize = 10
    leading = 12
    leftPadding = 6
    rightPadding = 6
    topPadding = 3
    bottomPadding = 3
    firstLineIndent = 0
    color = 'black'
    alignment = 'LEFT'
    background = 'white'
    valign = 'BOTTOM'
    href = None
    destination = None

    def __init__(self, name, parent=None):
        self.name = name
        if parent is not None:
            parent.copy(self)

    def copy(self, result=None):
        if result is None:
            result = CellStyle(self.name)
        for name in dir(self):
            if name.startswith('_'):
                continue
            setattr(result, name, getattr(self, name))
        return result