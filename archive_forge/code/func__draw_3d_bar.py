from reportlab.graphics.shapes import Drawing, Polygon, Line
from math import pi
def _draw_3d_bar(G, x1, x2, y0, yhigh, xdepth, ydepth, fillColor=None, fillColorShaded=None, strokeColor=None, strokeWidth=1, shading=0.1):
    fillColorShaded = _getShaded(fillColor, None, shading)
    fillColorShadedTop = _getShaded(fillColor, None, shading / 2.0)

    def _add_3d_bar(x1, x2, y1, y2, xoff, yoff, G=G, strokeColor=strokeColor, strokeWidth=strokeWidth, fillColor=fillColor):
        G.add(Polygon((x1, y1, x1 + xoff, y1 + yoff, x2 + xoff, y2 + yoff, x2, y2), strokeWidth=strokeWidth, strokeColor=strokeColor, fillColor=fillColor, strokeLineJoin=1))
    usd = max(y0, yhigh)
    if xdepth or ydepth:
        if y0 != yhigh:
            _add_3d_bar(x2, x2, y0, yhigh, xdepth, ydepth, fillColor=fillColorShaded)
        _add_3d_bar(x1, x2, usd, usd, xdepth, ydepth, fillColor=fillColorShadedTop)
    G.add(Polygon((x1, y0, x2, y0, x2, yhigh, x1, yhigh), strokeColor=strokeColor, strokeWidth=strokeWidth, fillColor=fillColor, strokeLineJoin=1))
    if xdepth or ydepth:
        G.add(Line(x1, usd, x2, usd, strokeWidth=strokeWidth, strokeColor=strokeColor or fillColorShaded))