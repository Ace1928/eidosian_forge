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
def _splitCell(self, value, style, oldHeight, newHeight, width):
    height0 = newHeight - style.topPadding
    height1 = oldHeight - (style.topPadding + newHeight)
    if isinstance(value, (tuple, list)):
        newCellContent = []
        postponedContent = []
        split = False
        cellHeight = self._listCellGeom(value, width, style)[1]
        if style.valign == 'MIDDLE':
            usedHeight = (oldHeight - cellHeight) / 2
        else:
            usedHeight = 0
        for flowable in value:
            if split:
                if flowable.height <= height1:
                    postponedContent.append(flowable)
                    height1 -= flowable.height
                else:
                    return []
            elif usedHeight + flowable.height <= height0:
                newCellContent.append(flowable)
                usedHeight += flowable.height
            else:
                splits = flowable.split(width, height0 - usedHeight)
                if splits:
                    newCellContent.append(splits[0])
                    postponedContent.append(splits[1])
                elif newCellContent or style.valign != 'TOP':
                    if flowable.height <= height1:
                        postponedContent.append(flowable)
                        height1 -= flowable.height
                    else:
                        return []
                else:
                    return []
                split = True
        return (tuple(newCellContent), tuple(postponedContent))
    elif isinstance(value, str):
        rows = value.split('\n')
        lineHeight = 1.2 * style.fontsize
        contentHeight = (style.leading or lineHeight) * len(rows)
        if style.valign == 'TOP' and contentHeight <= height0:
            return (value, '')
        elif style.valign == 'BOTTOM' and contentHeight <= height1:
            return ('', value)
        elif style.valign == 'MIDDLE':
            if height1 > height0:
                return ('', value)
            else:
                return (value, '')
        elif len(rows) < 2:
            return []
        if style.valign == 'TOP':
            splitPoint = height0 // lineHeight
        elif style.valign == 'BOTTOM':
            splitPoint = len(rows) - height1 // lineHeight
        else:
            splitPoint = (height0 - height1 + contentHeight) // (2 * lineHeight)
        splitPoint = int(splitPoint)
        return ('\n'.join(rows[:splitPoint]), '\n'.join(rows[splitPoint:]))
    return ('', '')