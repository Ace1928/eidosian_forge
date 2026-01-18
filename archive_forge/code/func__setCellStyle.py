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
def _setCellStyle(cellStyles, i, j, op, values):
    new = cellStyles[i][j]
    if op == 'FONT':
        n = len(values)
        new.fontname = values[0]
        if n > 1:
            new.fontsize = values[1]
            if n > 2:
                new.leading = values[2]
            else:
                new.leading = new.fontsize * 1.2
    elif op in ('FONTNAME', 'FACE'):
        new.fontname = values[0]
    elif op in ('SIZE', 'FONTSIZE'):
        new.fontsize = values[0]
    elif op == 'LEADING':
        new.leading = values[0]
    elif op == 'TEXTCOLOR':
        new.color = colors.toColor(values[0], colors.Color(0, 0, 0))
    elif op in ('ALIGN', 'ALIGNMENT'):
        new.alignment = values[0]
    elif op == 'VALIGN':
        new.valign = values[0]
    elif op == 'LEFTPADDING':
        new.leftPadding = values[0]
    elif op == 'RIGHTPADDING':
        new.rightPadding = values[0]
    elif op == 'TOPPADDING':
        new.topPadding = values[0]
    elif op == 'BOTTOMPADDING':
        new.bottomPadding = values[0]
    elif op == 'HREF':
        new.href = values[0]
    elif op == 'DESTINATION':
        new.destination = values[0]