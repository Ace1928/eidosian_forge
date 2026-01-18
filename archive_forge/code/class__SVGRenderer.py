import math, sys, os, codecs, base64
from io import BytesIO, StringIO
from reportlab.pdfbase.pdfmetrics import stringWidth # for font info
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.utils import asNative
from reportlab.graphics.renderbase import getStateDelta, Renderer, renderScaledDrawing
from reportlab.graphics.shapes import STATE_DEFAULTS, Path, UserNode
from reportlab.graphics.shapes import * # (only for test0)
from reportlab import rl_config
from reportlab.lib.utils import RLString, isUnicode, isBytes
from reportlab.pdfgen.canvas import FILL_EVEN_ODD, FILL_NON_ZERO
from .renderPM import _getImage
from xml.dom import getDOMImplementation
class _SVGRenderer(Renderer):
    """This draws onto an SVG document.
    """

    def __init__(self):
        self.verbose = 0

    def drawNode(self, node):
        """This is the recursive method called for each node in the tree.
        """
        if self.verbose:
            print('### begin _SVGRenderer.drawNode(%r)' % node)
        self._canvas.comment('begin node %r' % node)
        style = self._canvas.style.copy()
        if not (isinstance(node, Path) and node.isClipPath):
            pass
        deltas = getStateDelta(node)
        self._tracker.push(deltas)
        self.applyStateChanges(deltas, {})
        self.drawNodeDispatcher(node)
        rDeltas = self._tracker.pop()
        if not (isinstance(node, Path) and node.isClipPath):
            pass
        self._canvas.comment('end node %r' % node)
        for k, v in rDeltas.items():
            if k in self._restores:
                setattr(self._canvas, self._restores[k], v)
        self._canvas.style = style
        if self.verbose:
            print('### end _SVGRenderer.drawNode(%r)' % node)
    _restores = {'strokeColor': '_strokeColor', 'strokeWidth': '_lineWidth', 'strokeLineCap': '_lineCap', 'strokeLineJoin': '_lineJoin', 'fillColor': '_fillColor', 'fontName': '_font', 'fontSize': '_fontSize'}

    def _get_link_info_dict(self, obj):
        url = getattr(obj, 'hrefURL', '') or ''
        title = getattr(obj, 'hrefTitle', '') or ''
        if url:
            return {'xlink:href': url, 'xlink:title': title, 'target': '_top'}
        else:
            return None

    def drawGroup(self, group):
        if self.verbose:
            print('### begin _SVGRenderer.drawGroup')
        currGroup = self._canvas.startGroup()
        a, b, c, d, e, f = self._tracker.getState()['transform']
        for childNode in group.getContents():
            if isinstance(childNode, UserNode):
                node2 = childNode.provideNode()
            else:
                node2 = childNode
            self.drawNode(node2)
        self._canvas.transform(a, b, c, d, e, f)
        self._canvas.endGroup(currGroup)
        if self.verbose:
            print('### end _SVGRenderer.drawGroup')

    def drawRect(self, rect):
        link_info = self._get_link_info_dict(rect)
        svgAttrs = getattr(rect, '_svgAttrs', {})
        if rect.rx == rect.ry == 0:
            self._canvas.rect(rect.x, rect.y, rect.x + rect.width, rect.y + rect.height, link_info=link_info, **svgAttrs)
        else:
            self._canvas.roundRect(rect.x, rect.y, rect.x + rect.width, rect.y + rect.height, rect.rx, rect.ry, link_info=link_info, **svgAttrs)

    def drawString(self, stringObj):
        S = self._tracker.getState()
        text_anchor, x, y, text = (S['textAnchor'], stringObj.x, stringObj.y, stringObj.text)
        self._canvas.drawString(text, x, y, link_info=self._get_link_info_dict(stringObj), text_anchor=text_anchor, textRenderMode=getattr(stringObj, 'textRenderMode', 0), **getattr(stringObj, '_svgAttrs', {}))

    def drawLine(self, line):
        if self._canvas._strokeColor:
            self._canvas.line(line.x1, line.y1, line.x2, line.y2)

    def drawCircle(self, circle):
        self._canvas.circle(circle.cx, circle.cy, circle.r, link_info=self._get_link_info_dict(circle))

    def drawWedge(self, wedge):
        yradius, radius1, yradius1 = wedge._xtraRadii()
        if (radius1 == 0 or radius1 is None) and (yradius1 == 0 or yradius1 is None) and (not wedge.annular):
            centerx, centery, radius, startangledegrees, endangledegrees = (wedge.centerx, wedge.centery, wedge.radius, wedge.startangledegrees, wedge.endangledegrees)
            yradius = wedge.yradius or wedge.radius
            x1, y1 = (centerx - radius, centery - yradius)
            x2, y2 = (centerx + radius, centery + yradius)
            extent = endangledegrees - startangledegrees
            self._canvas.drawArc(x1, y1, x2, y2, startangledegrees, extent, fromcenter=1)
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
        self._canvas.ellipse(x1, y1, x2, y2, link_info=self._get_link_info_dict(ellipse))

    def drawPolygon(self, p):
        self._canvas.polygon(_pointsFromList(p.points), closed=1, link_info=self._get_link_info_dict(p))

    def drawPath(self, path, fillMode=FILL_EVEN_ODD):
        from reportlab.graphics.shapes import _renderPath
        c = self._canvas
        drawFuncs = (c.moveTo, c.lineTo, c.curveTo, c.closePath)
        if fillMode is None:
            fillMode = getattr(path, 'fillMode', FILL_EVEN_ODD)
        link_info = self._get_link_info_dict(path)
        autoclose = getattr(path, 'autoclose', '')

        def rP(**kwds):
            return _renderPath(path, drawFuncs, **kwds)
        if autoclose == 'svg':
            rP()
            c._fillAndStroke([], clip=path.isClipPath, link_info=link_info, fillMode=fillMode)
        elif autoclose == 'pdf':
            rP(forceClose=True)
            c._fillAndStroke([], clip=path.isClipPath, link_info=link_info, fillMode=fillMode)
        else:
            isClosed = rP()
            if not isClosed:
                ofc = c._fillColor
                c.setFillColor(None)
                try:
                    link_info = None
                    c._fillAndStroke([], clip=path.isClipPath, link_info=link_info, fillMode=fillMode)
                finally:
                    c.setFillColor(ofc)
            else:
                c._fillAndStroke([], clip=path.isClipPath, link_info=link_info, fillMode=fillMode)

    def drawImage(self, image):
        path = image.path
        if isinstance(path, str):
            if not (path and os.path.isfile(path)):
                return
            im = _getImage().open(path)
        elif hasattr(path, 'convert'):
            im = path
        else:
            return
        srcW, srcH = im.size
        dstW, dstH = (image.width, image.height)
        if dstW is None:
            dstW = srcW
        if dstH is None:
            dstH = srcH
        self._canvas.drawImage(im, image.x, image.y, dstW, dstH, embed=True)

    def applyStateChanges(self, delta, newState):
        """This takes a set of states, and outputs the operators
        needed to set those properties"""
        for key, value in delta.items():
            if key == 'transform':
                pass
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
            elif key == 'fillMode':
                self._canvas.setFillMode(value)