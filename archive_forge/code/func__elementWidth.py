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
def _elementWidth(self, v, s):
    if isinstance(v, (list, tuple)):
        w = 0
        for e in v:
            ew = self._elementWidth(e, s)
            if ew is None:
                return None
            w = max(w, ew)
        return w
    elif isinstance(v, Flowable):
        if v._fixedWidth:
            if hasattr(v, 'width') and isinstance(v.width, (int, float)):
                return v.width
            if hasattr(v, 'drawWidth') and isinstance(v.drawWidth, (int, float)):
                return v.drawWidth
        if hasattr(v, '__styledWrap__'):
            try:
                return getattr(v, '__styledWrap__')(s)[0]
            except:
                pass
    if hasattr(v, 'minWidth'):
        try:
            w = v.minWidth()
            if isinstance(w, (float, int)):
                return w
        except AttributeError:
            pass
    if v is None:
        return 0
    else:
        try:
            v = str(v).split('\n')
        except:
            return 0
    fontName = s.fontname
    fontSize = s.fontsize
    return max([stringWidth(x, fontName, fontSize) for x in v])