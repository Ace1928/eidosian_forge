from reportlab.pdfbase.pdfmetrics import getFont, unicode2T1
from reportlab.lib.utils import open_and_read, isBytes, rl_exec
from .shapes import _baseGFontName, _PATH_OP_ARG_COUNT, _PATH_OP_NAMES, definePath
from sys import exc_info
def _text2Path(self, text, x=0, y=0, fontName=_baseGFontName, fontSize=1000, **kwds):
    face, font = self.setFont(fontName)
    scale = fontSize / face.units_per_EM
    __dx__ = x / scale
    __dy__ = y / scale
    P = []
    S = []
    P_append = P.append
    truncate = kwds.pop('truncate', 0)
    if truncate:
        xpt = lambda x: pathNumTrunc(scale * (x + __dx__))
        ypt = lambda y: pathNumTrunc(scale * (y + __dy__))
    else:
        xpt = lambda x: scale * (x + __dx__)
        ypt = lambda y: scale * (y + __dy__)

    def move_to(a, ctx):
        if P:
            P_append(('closePath',))
        P_append(('moveTo', xpt(a.x), ypt(a.y)))

    def line_to(a, ctx):
        P_append(('lineTo', xpt(a.x), ypt(a.y)))

    def conic_to(a, b, ctx):
        """using the cubic equivalent"""
        x0, y0 = P[-1][-2:] if P else (a.x, a.y)
        x1 = xpt(a.x)
        y1 = ypt(a.y)
        x2 = xpt(b.x)
        y2 = ypt(b.y)
        P_append(('curveTo', x0 + (x1 - x0) * 2 / 3, y0 + (y1 - y0) * 2 / 3, x1 + (x2 - x1) / 3, y1 + (y2 - y1) / 3, x2, y2))

    def cubic_to(a, b, c, ctx):
        P_append(('curveTo', xpt(a.x), ypt(a.y), xpt(b.x), ypt(b.y), xpt(c.x), ypt(c.y)))
    lineHeight = fontSize * 1.2 / scale
    ftLFlags = self.ftLFlags
    for c in text:
        if c == '\n':
            __dx__ = 0
            __dy__ -= lineHeight
            continue
        face.load_char(c, ftLFlags)
        face.glyph.outline.decompose(self, move_to=move_to, line_to=line_to, conic_to=conic_to, cubic_to=cubic_to)
        __dx__ = __dx__ + face.glyph.metrics.horiAdvance
    if P:
        P_append(('closePath',))
    return P