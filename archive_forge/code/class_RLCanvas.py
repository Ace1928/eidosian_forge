import os
import types
from math import *
from reportlab.graphics import shapes
from reportlab.lib import colors
from rdkit.sping.PDF import pdfmetrics, pidPDF
from rdkit.sping.pid import *
class RLCanvas(Canvas):

    def __init__(self, size=(300, 300), name='RLCanvas'):
        self.size = size
        self._initOutput()
        Canvas.__init__(self, size, name)
        self.drawing = shapes.Drawing(size[0], size[1])

    def _initOutput(self):
        pass

    def clear(self):
        self._initOutput()

    def flush(self):
        pass

    def save(self, file=None, format=None):
        """Hand this either a file= <filename> or
    file = <an open file object>.  
    """
        if not file:
            file = self.name
        from reportlab.graphics import renderPDF
        renderPDF.drawToFile(self.drawing, file, self.name)

    def fixY(self, y):
        return self.size[1] - y

    def _findPostScriptFontName(self, font):
        """Attempts to return proper font name."""
        if not font.face:
            face = 'serif'
        else:
            face = font.face.lower()
        while face in pidPDF.font_face_map:
            face = pidPDF.font_face_map[face]
        psname = pidPDF.ps_font_map[face, font.bold, font.italic]
        return psname

    def drawLine(self, x1, y1, x2, y2, color=None, width=None, dash=None, **kwargs):
        """Draw a straight line between x1,y1 and x2,y2."""
        if color:
            if color == transparent:
                return
        elif self.defaultLineColor == transparent:
            return
        else:
            color = self.defaultLineColor
        color = colorToRL(color)
        if width:
            w = width
        else:
            w = self.defaultLineWidth
        self.drawing.add(shapes.Line(x1, self.fixY(y1), x2, self.fixY(y2), strokeColor=color, strokeWidth=w, strokeDashArray=dash))
        return

    def drawCurve(self, x1, y1, x2, y2, x3, y3, x4, y4, closed=0, **kwargs):
        """Draw a Bezier curve with control points x1,y1 to x4,y4."""
        pts = self.curvePoints(x1, y1, x2, y2, x3, y3, x4, y4)
        if not closed:
            pointlist = [(pts[x][0], pts[x][1], pts[x + 1][0], pts[x + 1][1]) for x in range(len(pts) - 1)]
            self.drawLines(pointlist, **kwargs)
        else:
            self.drawPolygon(pointlist, closed=1, **kwargs)

    def drawArc(self, x1, y1, x2, y2, startAng=0, extent=360, edgeColor=None, edgeWidth=None, fillColor=None, dash=None, **kwargs):
        """Draw a partial ellipse inscribed within the rectangle x1,y1,x2,y2, 
    starting at startAng degrees and covering extent degrees.   Angles 
    start with 0 to the right (+x) and increase counter-clockwise. 
    These should have x1<x2 and y1<y2."""
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        pointlist = self.arcPoints(x1, y1, x2, y2, startAng, extent)
        self.drawPolygon(pointlist + [center], edgeColor=transparent, edgeWidth=0, fillColor=fillColor)
        pts = pointlist
        pointlist = [(pts[x][0], pts[x][1], pts[x + 1][0], pts[x + 1][1]) for x in range(len(pts) - 1)]
        self.drawLines(pointlist, edgeColor, edgeWidth, dash=dash, **kwargs)

    def drawPolygon(self, pointlist, edgeColor=None, edgeWidth=None, fillColor=transparent, closed=0, dash=None, **kwargs):
        """drawPolygon(pointlist) -- draws a polygon
    pointlist: a list of (x,y) tuples defining vertices
    """
        if not edgeColor:
            edgeColor = self.defaultLineColor
        edgeColor = colorToRL(edgeColor)
        if not fillColor or fillColor == transparent:
            fillColor = None
        else:
            fillColor = colorToRL(fillColor)
        if edgeWidth:
            w = edgeWidth
        else:
            w = self.defaultLineWidth
        points = []
        for x, y in pointlist:
            points.append(x)
            points.append(self.fixY(y))
        self.drawing.add(shapes.Polygon(points, strokeColor=edgeColor, strokeWidth=w, strokeDashArray=dash, fillColor=fillColor))

    def drawString(self, s, x, y, font=None, color=None, angle=0, **kwargs):
        if color:
            if color == transparent:
                return
        elif self.defaultLineColor == transparent:
            return
        else:
            color = self.defaultLineColor
        color = colorToRL(color)
        if font is None:
            font = self.defaultFont
        txt = shapes.String(0, 0, s, fillColor=color)
        txt.fontName = self._findPostScriptFontName(font)
        txt.fontSize = font.size
        g = shapes.Group(txt)
        g.translate(x, self.fixY(y))
        g.rotate(angle)
        self.drawing.add(g)
        return

    def drawImage(self, image, x1, y1, x2=None, y2=None, **kwargs):
        """
      to the best of my knowledge, the only real way to get an image
    """
        return

    def stringWidth(self, s, font=None):
        """Return the logical width of the string if it were drawn     in the current font (defaults to self.font)."""
        if not font:
            font = self.defaultFont
        fontName = self._findPostScriptFontName(font)
        return pdfmetrics.stringwidth(s, fontName) * font.size * 0.001

    def fontAscent(self, font=None):
        if not font:
            font = self.defaultFont
        fontName = self._findPostScriptFontName(font)
        return pdfmetrics.ascent_descent[fontName][0] * 0.001 * font.size

    def fontDescent(self, font=None):
        if not font:
            font = self.defaultFont
        fontName = self._findPostScriptFontName(font)
        return -pdfmetrics.ascent_descent[fontName][1] * 0.001 * font.size