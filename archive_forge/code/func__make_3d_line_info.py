from reportlab.graphics.shapes import Drawing, Polygon, Line
from math import pi
def _make_3d_line_info(G, x0, x1, y0, y1, z0, z1, theta_x, theta_y, fillColor, fillColorShaded=None, tileWidth=1, strokeColor=None, strokeWidth=None, strokeDashArray=None, shading=0.1):
    zwidth = abs(z1 - z0)
    xdepth = zwidth * theta_x
    ydepth = zwidth * theta_y
    depth_slope = xdepth == 0 and 1e+150 or -ydepth / float(xdepth)
    x = float(x1 - x0)
    slope = x == 0 and 1e+150 or (y1 - y0) / x
    c = slope > depth_slope and _getShaded(fillColor, fillColorShaded, shading) or fillColor
    zy0 = z0 * theta_y
    zx0 = z0 * theta_x
    tileStrokeWidth = 0.6
    if tileWidth is None:
        D = [(x1, y1)]
    else:
        T = ((y1 - y0) ** 2 + (x1 - x0) ** 2) ** 0.5
        tileStrokeWidth *= tileWidth
        if T < tileWidth:
            D = [(x1, y1)]
        else:
            n = int(T / float(tileWidth)) + 1
            dx = float(x1 - x0) / n
            dy = float(y1 - y0) / n
            D = []
            a = D.append
            for i in range(1, n):
                a((x0 + dx * i, y0 + dy * i))
    a = G.add
    x_0 = x0 + zx0
    y_0 = y0 + zy0
    for x, y in D:
        x_1 = x + zx0
        y_1 = y + zy0
        P = Polygon(_ystrip_poly(x_0, x_1, y_0, y_1, xdepth, ydepth), fillColor=c, strokeColor=c, strokeWidth=tileStrokeWidth)
        a((0, z0, z1, x_0, y_0, P))
        x_0 = x_1
        y_0 = y_1