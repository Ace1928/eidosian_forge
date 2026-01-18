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
def _quadrantDef(xpos, ypos, corner, r, kind=0, direction='left-right', m=0.4472):
    t = m * r
    if xpos == 'right' and ypos == 'bottom':
        xhi, ylo = corner
        P = [(xhi - r, ylo), (xhi - t, ylo), (xhi, ylo + t), (xhi, ylo + r)]
    elif xpos == 'right' and ypos == 'top':
        xhi, yhi = corner
        P = [(xhi, yhi - r), (xhi, yhi - t), (xhi - t, yhi), (xhi - r, yhi)]
    elif xpos == 'left' and ypos == 'top':
        xlo, yhi = corner
        P = [(xlo + r, yhi), (xlo + t, yhi), (xlo, yhi - t), (xlo, yhi - r)]
    elif xpos == 'left' and ypos == 'bottom':
        xlo, ylo = corner
        P = [(xlo, ylo + r), (xlo, ylo + t), (xlo + t, ylo), (xlo + r, ylo)]
    else:
        raise ValueError(f'Unknown quadrant position (xpos,ypos)={(xpos, ypos)!r}')
    if direction == 'left-right' and P[0][0] > P[-1][0] or (direction == 'bottom-top' and P[0][1] > P[-1][1]):
        P.reverse()
    P = _calcBezierPoints(P, kind)
    return P