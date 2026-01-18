import math
from io import BytesIO, StringIO
from reportlab.pdfbase.pdfmetrics import getFont, stringWidth, unicode2T1 # for font info
from reportlab.lib.utils import asBytes, char2int, rawBytes, asNative, isUnicode
from reportlab.lib.rl_accel import fp_str
from reportlab.graphics.renderbase import Renderer, getStateDelta, renderScaledDrawing
from reportlab.graphics.shapes import STATE_DEFAULTS
from reportlab import rl_config
from reportlab.pdfgen.canvas import FILL_EVEN_ODD
from reportlab.graphics.shapes import *
class _PSRenderer(Renderer):
    """This draws onto a EPS document.  It needs to be a class
    rather than a function, as some EPS-specific state tracking is
    needed outside of the state info in the SVG model."""

    def drawNode(self, node):
        """This is the recursive method called for each node
        in the tree"""
        self._canvas.comment('begin node %r' % node)
        color = self._canvas._color
        if not (isinstance(node, Path) and node.isClipPath):
            self._canvas.saveState()
        deltas = getStateDelta(node)
        self._tracker.push(deltas)
        self.applyStateChanges(deltas, {})
        self.drawNodeDispatcher(node)
        rDeltas = self._tracker.pop()
        if not (isinstance(node, Path) and node.isClipPath):
            self._canvas.restoreState()
        self._canvas.comment('end node %r' % node)
        self._canvas._color = color
        for k, v in rDeltas.items():
            if k in self._restores:
                setattr(self._canvas, self._restores[k], v)
    _restores = {'strokeColor': '_strokeColor', 'strokeWidth': '_lineWidth', 'strokeLineCap': '_lineCap', 'strokeLineJoin': '_lineJoin', 'fillColor': '_fillColor', 'fontName': '_font', 'fontSize': '_fontSize'}

    def drawRect(self, rect):
        if rect.rx == rect.ry == 0:
            self._canvas.rect(rect.x, rect.y, rect.x + rect.width, rect.y + rect.height)
        else:
            self._canvas.roundRect(rect.x, rect.y, rect.x + rect.width, rect.y + rect.height, rect.rx, rect.ry)

    def drawLine(self, line):
        if self._canvas._strokeColor:
            self._canvas.line(line.x1, line.y1, line.x2, line.y2)

    def drawCircle(self, circle):
        self._canvas.circle(circle.cx, circle.cy, circle.r)

    def drawWedge(self, wedge):
        yradius, radius1, yradius1 = wedge._xtraRadii()
        if (radius1 == 0 or radius1 is None) and (yradius1 == 0 or yradius1 is None) and (not wedge.annular):
            startangledegrees = wedge.startangledegrees
            endangledegrees = wedge.endangledegrees
            centerx = wedge.centerx
            centery = wedge.centery
            radius = wedge.radius
            extent = endangledegrees - startangledegrees
            self._canvas.drawArc(centerx - radius, centery - yradius, centerx + radius, centery + yradius, startangledegrees, extent, fromcenter=1)
        else:
            P = wedge.asPolygon()
            if isinstance(P, Path):
                self.drawPath(P)
            else:
                self.drawPolygon(P)

    def drawPolyLine(self, p):
        if self._canvas._strokeColor:
            self._canvas.polyLine(_pointsFromList(p.points))

    def drawEllipse(self, ellipse):
        x1 = ellipse.cx - ellipse.rx
        x2 = ellipse.cx + ellipse.rx
        y1 = ellipse.cy - ellipse.ry
        y2 = ellipse.cy + ellipse.ry
        self._canvas.ellipse(x1, y1, x2, y2)

    def drawPolygon(self, p):
        self._canvas.polygon(_pointsFromList(p.points), closed=1)

    def drawString(self, stringObj):
        textRenderMode = getattr(stringObj, 'textRenderMode', 0)
        if self._canvas._fillColor or textRenderMode:
            S = self._tracker.getState()
            text_anchor, x, y, text = (S['textAnchor'], stringObj.x, stringObj.y, stringObj.text)
            if not text_anchor in ['start', 'inherited']:
                font, fontSize = (S['fontName'], S['fontSize'])
                textLen = stringWidth(text, font, fontSize)
                if text_anchor == 'end':
                    x -= textLen
                elif text_anchor == 'middle':
                    x -= textLen / 2
                elif text_anchor == 'numeric':
                    x -= numericXShift(text_anchor, text, textLen, font, fontSize, encoding='winansi')
                else:
                    raise ValueError('bad value for text_anchor ' + str(text_anchor))
            self._canvas.drawString(x, y, text, textRenderMode=textRenderMode)

    def drawPath(self, path, fillMode=None):
        from reportlab.graphics.shapes import _renderPath
        c = self._canvas
        drawFuncs = (c.moveTo, c.lineTo, c.curveTo, c.closePath)
        autoclose = getattr(path, 'autoclose', '')

        def rP(**kwds):
            return _renderPath(path, drawFuncs, **kwds)
        if fillMode is None:
            fillMode = getattr(path, 'fillMode', c._fillMode)
        fill = c._fillColor is not None
        stroke = c._strokeColor is not None
        clip = path.isClipPath
        fas = lambda **kwds: c._fillAndStroke([], fillMode=fillMode, **kwds)
        pathFill = lambda: c._fillAndStroke([], stroke=0, fillMode=fillMode)
        pathStroke = lambda: c._fillAndStroke([], fill=0)
        if autoclose == 'svg':
            rP()
            fas(stroke=stroke, fill=fill, clip=clip)
        elif autoclose == 'pdf':
            if fill:
                rP(forceClose=True)
                fas(stroke=stroke, fill=fill, clip=clip)
            elif stroke or clip:
                rP()
                fas(stroke=stroke, fill=0, clip=clip)
        elif fill and rP(countOnly=True):
            rP()
        elif stroke or clip:
            rP()
            fas(stroke=stroke, fill=0, clip=clip)

    def applyStateChanges(self, delta, newState):
        """This takes a set of states, and outputs the operators
        needed to set those properties"""
        for key, value in delta.items():
            if key == 'transform':
                self._canvas.transform(value[0], value[1], value[2], value[3], value[4], value[5])
            elif key == 'strokeColor':
                self._canvas.setStrokeColor(value)
            elif key == 'strokeWidth':
                self._canvas.setLineWidth(value)
            elif key == 'strokeLineCap':
                self._canvas.setLineCap(value)
            elif key == 'strokeLineJoin':
                self._canvas.setLineJoin(value)
            elif key == 'strokeDashArray':
                if value:
                    if isinstance(value, (list, tuple)) and len(value) == 2 and isinstance(value[1], (tuple, list)):
                        phase = value[0]
                        value = value[1]
                    else:
                        phase = 0
                    self._canvas.setDash(value, phase)
                else:
                    self._canvas.setDash()
            elif key == 'fillColor':
                self._canvas.setFillColor(value)
            elif key in ['fontSize', 'fontName']:
                fontname = delta.get('fontName', self._canvas._font)
                fontsize = delta.get('fontSize', self._canvas._fontSize)
                self._canvas.setFont(fontname, fontsize)

    def drawImage(self, image):
        from reportlab.lib.utils import ImageReader
        im = ImageReader(image.path)
        self._canvas.drawImage(im._image, image.x, image.y, image.width, image.height)