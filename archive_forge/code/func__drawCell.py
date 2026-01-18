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
def _drawCell(self, cellval, cellstyle, pos, size):
    colpos, rowpos = pos
    colwidth, rowheight = size
    if self._curcellstyle is not cellstyle:
        cur = self._curcellstyle
        if cur is None or cellstyle.color != cur.color:
            self.canv.setFillColor(cellstyle.color)
        if cur is None or cellstyle.leading != cur.leading or cellstyle.fontname != cur.fontname or (cellstyle.fontsize != cur.fontsize):
            self.canv.setFont(cellstyle.fontname, cellstyle.fontsize, cellstyle.leading)
        self._curcellstyle = cellstyle
    just = cellstyle.alignment
    valign = cellstyle.valign
    if isinstance(cellval, (tuple, list, Flowable)):
        if not isinstance(cellval, (tuple, list)):
            cellval = (cellval,)
        W = []
        H = []
        w, h = self._listCellGeom(cellval, colwidth, cellstyle, W=W, H=H, aH=rowheight)
        if valign == 'TOP':
            y = rowpos + rowheight - cellstyle.topPadding
        elif valign == 'BOTTOM':
            y = rowpos + cellstyle.bottomPadding + h
        else:
            y = rowpos + (rowheight + cellstyle.bottomPadding - cellstyle.topPadding + h) / 2.0
        if cellval:
            y += cellval[0].getSpaceBefore()
        for v, w, h in zip(cellval, W, H):
            if just == 'LEFT':
                x = colpos + cellstyle.leftPadding
            elif just == 'RIGHT':
                x = colpos + colwidth - cellstyle.rightPadding - w
            elif just in ('CENTRE', 'CENTER'):
                x = colpos + (colwidth + cellstyle.leftPadding - cellstyle.rightPadding - w) / 2.0
            else:
                raise ValueError(f'Invalid justification {just!a} for {type(v)}')
            y -= v.getSpaceBefore()
            y -= h
            v.drawOn(self.canv, x, y)
            y -= v.getSpaceAfter()
    else:
        if just == 'LEFT':
            draw = self.canv.drawString
            x = colpos + cellstyle.leftPadding
        elif just in ('CENTRE', 'CENTER'):
            draw = self.canv.drawCentredString
            x = colpos + (colwidth + cellstyle.leftPadding - cellstyle.rightPadding) * 0.5
        elif just == 'RIGHT':
            draw = self.canv.drawRightString
            x = colpos + colwidth - cellstyle.rightPadding
        elif just == 'DECIMAL':
            draw = self.canv.drawAlignedString
            x = colpos + colwidth - cellstyle.rightPadding
        else:
            raise ValueError(f'Invalid justification {just!a}')
        vals = str(cellval).split('\n')
        n = len(vals)
        leading = cellstyle.leading
        fontsize = cellstyle.fontsize
        if valign == 'BOTTOM':
            y = rowpos + cellstyle.bottomPadding + n * leading - fontsize
        elif valign == 'TOP':
            y = rowpos + rowheight - cellstyle.topPadding - fontsize
        elif valign == 'MIDDLE':
            y = rowpos + (cellstyle.bottomPadding + rowheight - cellstyle.topPadding + n * leading) / 2.0 - fontsize
        else:
            raise ValueError(f'Bad valign: {valign!a}')
        for v in vals:
            draw(x, y, v)
            y -= leading
        onDraw = getattr(cellval, 'onDraw', None)
        if onDraw:
            onDraw(self.canv, cellval.kind, cellval.label)
    if cellstyle.href:
        self.canv.linkURL(cellstyle.href, (colpos, rowpos, colpos + colwidth, rowpos + rowheight), relative=1)
    if cellstyle.destination:
        self.canv.linkRect('', cellstyle.destination, Rect=(colpos, rowpos, colpos + colwidth, rowpos + rowheight), relative=1)