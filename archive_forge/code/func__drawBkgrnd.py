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
def _drawBkgrnd(self):
    nrows = self._nrows
    ncols = self._ncols
    canv = self.canv
    colpositions = self._colpositions
    rowpositions = self._rowpositions
    rowHeights = self._rowHeights
    colWidths = self._colWidths
    spanRects = getattr(self, '_spanRects', None)
    for cmd, (sc, sr), (ec, er), arg in self._bkgrndcmds:
        if sr in _SPECIALROWS:
            continue
        if sc < 0:
            sc = sc + ncols
        if ec < 0:
            ec = ec + ncols
        if sr < 0:
            sr = sr + nrows
        if er < 0:
            er = er + nrows
        x0 = colpositions[sc]
        y0 = rowpositions[sr]
        x1 = colpositions[min(ec + 1, ncols)]
        y1 = rowpositions[min(er + 1, nrows)]
        w, h = (x1 - x0, y1 - y0)
        if hasattr(arg, '__call__'):
            arg(self, canv, x0, y0, w, h)
        elif cmd == 'ROWBACKGROUNDS':
            colorCycle = list(map(colors.toColorOrNone, arg))
            count = len(colorCycle)
            rowCount = er - sr + 1
            for i in range(rowCount):
                color = colorCycle[i % count]
                h = rowHeights[sr + i]
                if color:
                    canv.setFillColor(color)
                    canv.rect(x0, y0, w, -h, stroke=0, fill=1)
                y0 = y0 - h
        elif cmd == 'COLBACKGROUNDS':
            colorCycle = list(map(colors.toColorOrNone, arg))
            count = len(colorCycle)
            colCount = ec - sc + 1
            for i in range(colCount):
                color = colorCycle[i % count]
                w = colWidths[sc + i]
                if color:
                    canv.setFillColor(color)
                    canv.rect(x0, y0, w, h, stroke=0, fill=1)
                x0 = x0 + w
        elif arg and isinstance(arg, (list, tuple)) and (arg[0] in ('VERTICAL', 'HORIZONTAL', 'VERTICAL2', 'HORIZONTAL2', 'LINEARGRADIENT', 'RADIALGRADIENT')):
            if ec == sc and er == sr and spanRects:
                xywh = spanRects.get((sc, sr))
                if xywh:
                    x0, y0, w, h = xywh
            arg0, arg = (arg[0], arg[1:])
            canv.saveState()
            p = canv.beginPath()
            p.rect(x0, y0, w, h)
            canv.clipPath(p, stroke=0)
            if arg0 == 'HORIZONTAL':
                canv.linearGradient(x0, y0, x0 + w, y0, arg, extend=False)
            elif arg0 == 'HORIZONTAL2':
                xh = x0 + w / 2.0
                canv.linearGradient(x0, y0, xh, y0, arg, extend=False)
                canv.linearGradient(xh, y0, x0 + w, y0, arg[::-1], extend=False)
            elif arg0 == 'VERTICAL2':
                yh = y0 + h / 2.0
                canv.linearGradient(x0, y0, x0, yh, arg, extend=False)
                canv.linearGradient(x0, yh, x0, y0 + h, arg[::-1], extend=False)
            elif arg0 == 'VERTICAL':
                canv.linearGradient(x0, y0, x0, y0 + h, arg, extend=False)
            elif arg0 == 'LINEARGRADIENT':
                if 4 <= len(arg) <= 5:
                    (ax0, ay0), (ax1, ay1) = arg[:2]
                    ax0 = x0 + ax0 * w
                    ax1 = x0 + ax1 * w
                    ay0 = y0 + ay0 * h
                    ay1 = y0 + ay1 * h
                    extend = arg[2]
                    C = arg[3]
                    P = arg[4] if len(arg) == 4 else None
                    canv.linearGradient(ax0, ay0, ax1, ay1, C, positions=P, extend=extend)
                else:
                    raise ValueError(f'Wrong length for {op!a} arguments {arg!a}')
            elif arg0 == 'RADIALGRADIENT':
                if 4 <= len(arg) <= 5:
                    xc, yc = arg[0]
                    xc = x0 + xc * w
                    yc = y0 + yc * h
                    r, ref = arg[1]
                    if ref == 'width':
                        ref = w
                    elif ref == 'height':
                        ref = h
                    elif ref == 'min':
                        ref = min(w, h)
                    elif ref == 'max':
                        ref = max(w, h)
                    else:
                        raise ValueError(f'Bad radius, {arg[1]!a}, for {op!a} arguments {arg!r}')
                    r *= ref
                    extend = arg[2]
                    C = arg[3]
                    P = arg[4] if len(arg) == 4 else None
                    canv.radialGradient(xc, yc, r, C, positions=P, extend=extend)
                else:
                    raise ValueError(f'Wrong length for {op!a} arguments {arg}')
            canv.restoreState()
        else:
            color = colors.toColorOrNone(arg)
            if color:
                if ec == sc and er == sr and spanRects:
                    xywh = spanRects.get((sc, sr))
                    if xywh:
                        x0, y0, w, h = xywh
                canv.setFillColor(color)
                canv.rect(x0, y0, w, h, stroke=0, fill=1)