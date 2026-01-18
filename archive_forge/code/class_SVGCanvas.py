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
class SVGCanvas:

    def __init__(self, size=(300, 300), encoding='utf-8', verbose=0, bom=False, **kwds):
        """
        verbose = 0 >0 means do verbose stuff
        useClip = False True means don't use a clipPath definition put the global clip into the clip property
                        to get around an issue with safari
        extraXmlDecl = ''   use to add extra xml declarations
        scaleGroupId = ''   id of an extra group to add around the drawing to allow easy scaling
        svgAttrs = {}       dictionary of attributes to be applied to the svg tag itself
        """
        self.verbose = verbose
        self.encoding = codecs.lookup(encoding).name
        self.bom = bom
        useClip = kwds.pop('useClip', False)
        self.fontHacks = kwds.pop('fontHacks', {})
        self.extraXmlDecl = kwds.pop('extraXmlDecl', '')
        scaleGroupId = kwds.pop('scaleGroupId', '')
        self._fillMode = FILL_EVEN_ODD
        self.width, self.height = self.size = size
        self.code = []
        self.style = {}
        self.path = ''
        self._strokeColor = self._fillColor = self._lineWidth = self._font = self._fontSize = self._lineCap = self._lineJoin = None
        if kwds.pop('use_fp_str', False):
            self.fp_str = fp_str
        else:
            self.fp_str = py_fp_str
        self.cfp_str = lambda *args: self.fp_str(*args).replace(' ', ',')
        implementation = getDOMImplementation('minidom')
        doctype = implementation.createDocumentType('svg', '-//W3C//DTD SVG 1.0//EN', 'http://www.w3.org/TR/2001/REC-SVG-20010904/DTD/svg10.dtd')
        self.doc = implementation.createDocument(None, 'svg', doctype)
        self.svg = self.doc.documentElement
        svgAttrs = dict(width=str(size[0]), height=str(self.height), preserveAspectRatio='xMinYMin meet', viewBox='0 0 %d %d' % (self.width, self.height), xmlns='http://www.w3.org/2000/svg', version='1.0')
        svgAttrs['fill-rule'] = _fillRuleMap[self._fillMode]
        svgAttrs['xmlns:xlink'] = 'http://www.w3.org/1999/xlink'
        svgAttrs.update(kwds.pop('svgAttrs', {}))
        for k, v in svgAttrs.items():
            self.svg.setAttribute(k, v)
        title = self.doc.createElement('title')
        text = self.doc.createTextNode('...')
        title.appendChild(text)
        self.svg.appendChild(title)
        desc = self.doc.createElement('desc')
        text = self.doc.createTextNode('...')
        desc.appendChild(text)
        self.svg.appendChild(desc)
        self.setFont(STATE_DEFAULTS['fontName'], STATE_DEFAULTS['fontSize'])
        self.setStrokeColor(STATE_DEFAULTS['strokeColor'])
        self.setLineCap(2)
        self.setLineJoin(0)
        self.setLineWidth(1)
        if not useClip:
            clipPath = transformNode(self.doc, 'clipPath', id='clip')
            clipRect = transformNode(self.doc, 'rect', x=0, y=0, width=self.width, height=self.height)
            clipPath.appendChild(clipRect)
            self.svg.appendChild(clipPath)
            gtkw = dict(style='clip-path: url(#clip)')
        else:
            gtkw = dict(clip='0 0 %d %d' % (self.width, self.height))
        self.groupTree = transformNode(self.doc, 'g', id='group', transform='scale(1,-1) translate(0,-%d)' % self.height, **gtkw)
        if scaleGroupId:
            self.scaleTree = transformNode(self.doc, 'g', id=scaleGroupId, transform='scale(1,1)')
            self.scaleTree.appendChild(self.groupTree)
            self.svg.appendChild(self.scaleTree)
        else:
            self.svg.appendChild(self.groupTree)
        self.currGroup = self.groupTree

    def save(self, fn=None):
        writer = EncodedWriter(self.encoding, bom=self.bom)
        self.doc.writexml(writer, addindent='\t', newl='\n', encoding=self.encoding)
        if hasattr(fn, 'write'):
            f = fn
        else:
            f = open(fn, 'w', encoding=self.encoding)
        svg = writer.getvalue()
        exd = self.extraXmlDecl
        if exd:
            svg = svg.replace('?>', '?>' + exd)
        f.write(svg)
        if f is not fn:
            f.close()

    def NOTUSED_stringWidth(self, s, font=None, fontSize=None):
        """Return the logical width of the string if it were drawn
        in the current font (defaults to self.font).
        """
        font = font or self._font
        fontSize = fontSize or self._fontSize
        return stringWidth(s, font, fontSize)

    def _formatStyle(self, include=[], exclude='', **kwds):
        style = self.style.copy()
        style.update(kwds)
        keys = list(style.keys())
        if include:
            keys = [k for k in keys if k in include]
        if exclude:
            exclude = exclude.split()
            items = [k + ': ' + str(style[k]) for k in keys if k not in exclude]
        else:
            items = [k + ': ' + str(style[k]) for k in keys]
        return '; '.join(items) + ';'

    def _escape(self, s):
        """I don't think this was ever needed; seems to have been copied from renderPS"""
        return s

    def _genArcCode(self, x1, y1, x2, y2, startAng, extent):
        """Calculate the path for an arc inscribed in rectangle defined
        by (x1,y1),(x2,y2)."""
        return
        xScale = abs((x2 - x1) / 2.0)
        yScale = abs((y2 - y1) / 2.0)
        x, y = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        codeline = 'matrix currentmatrix %s %s translate %s %s scale 0 0 1 %s %s %s setmatrix'
        if extent >= 0:
            arc = 'arc'
        else:
            arc = 'arcn'
        data = (x, y, xScale, yScale, startAng, startAng + extent, arc)
        return codeline % data

    def _fillAndStroke(self, code, clip=0, link_info=None, styles=AREA_STYLES, fillMode=None):
        xtra = {}
        if fillMode:
            xtra['fill-rule'] = _fillRuleMap[fillMode]
        path = transformNode(self.doc, 'path', d=self.path, style=self._formatStyle(styles))
        if link_info:
            path = self._add_link(path, link_info)
        self.currGroup.appendChild(path)
        self.path = ''

    def setLineCap(self, v):
        vals = {0: 'butt', 1: 'round', 2: 'square'}
        if self._lineCap != v:
            self._lineCap = v
            self.style['stroke-linecap'] = vals[v]

    def setLineJoin(self, v):
        vals = {0: 'miter', 1: 'round', 2: 'bevel'}
        if self._lineJoin != v:
            self._lineJoin = v
            self.style['stroke-linecap'] = vals[v]

    def setDash(self, array=[], phase=0):
        """Two notations. Pass two numbers, or an array and phase."""
        if isinstance(array, (float, int)):
            self.style['stroke-dasharray'] = ', '.join(map(str, [array, phase]))
        elif isinstance(array, (tuple, list)) and len(array) > 0:
            assert phase >= 0, 'phase is a length in user space'
            self.style['stroke-dasharray'] = ', '.join(map(str, array))
            if phase > 0:
                self.style['stroke-dashoffset'] = str(phase)

    def setStrokeColor(self, color):
        self._strokeColor = color
        if color == None:
            self.style['stroke'] = 'none'
        else:
            r, g, b = (color.red, color.green, color.blue)
            self.style['stroke'] = 'rgb(%d%%,%d%%,%d%%)' % (r * 100, g * 100, b * 100)
            alpha = color.normalizedAlpha
            if alpha != 1:
                self.style['stroke-opacity'] = '%s' % alpha
            elif 'stroke-opacity' in self.style:
                del self.style['stroke-opacity']

    def setFillColor(self, color):
        self._fillColor = color
        if color == None:
            self.style['fill'] = 'none'
        else:
            r, g, b = (color.red, color.green, color.blue)
            self.style['fill'] = 'rgb(%d%%,%d%%,%d%%)' % (r * 100, g * 100, b * 100)
            alpha = color.normalizedAlpha
            if alpha != 1:
                self.style['fill-opacity'] = '%s' % alpha
            elif 'fill-opacity' in self.style:
                del self.style['fill-opacity']

    def setFillMode(self, v):
        self._fillMode = v
        self.style['fill-rule'] = _fillRuleMap[v]

    def setLineWidth(self, width):
        if width != self._lineWidth:
            self._lineWidth = width
            self.style['stroke-width'] = width

    def setFont(self, font, fontSize):
        if self._font != font or self._fontSize != fontSize:
            self._font = font
            self._fontSize = fontSize
            style = self.style
            for k in TEXT_STYLES:
                if k in style:
                    del style[k]
            svgAttrs = self.fontHacks[font] if font in self.fontHacks else {}
            if isinstance(font, RLString):
                svgAttrs.update(iter(font.svgAttrs.items()))
            if svgAttrs:
                for k, v in svgAttrs.items():
                    a = 'font-' + k
                    if a in TEXT_STYLES:
                        style[a] = v
            if 'font-family' not in style:
                style['font-family'] = font
            style['font-size'] = '%spx' % fontSize

    def _add_link(self, dom_object, link_info):
        assert isinstance(link_info, dict)
        link = transformNode(self.doc, 'a', **link_info)
        link.appendChild(dom_object)
        return link

    def rect(self, x1, y1, x2, y2, rx=8, ry=8, link_info=None, **_svgAttrs):
        """Draw a rectangle between x1,y1 and x2,y2."""
        if self.verbose:
            print('+++ SVGCanvas.rect')
        x = min(x1, x2)
        y = min(y1, y2)
        kwds = {}
        rect = transformNode(self.doc, 'rect', x=x, y=y, width=max(x1, x2) - x, height=max(y1, y2) - y, style=self._formatStyle(AREA_STYLES), **_svgAttrs)
        if link_info:
            rect = self._add_link(rect, link_info)
        self.currGroup.appendChild(rect)

    def roundRect(self, x1, y1, x2, y2, rx=8, ry=8, link_info=None, **_svgAttrs):
        """Draw a rounded rectangle between x1,y1 and x2,y2.

        Corners inset as ellipses with x-radius rx and y-radius ry.
        These should have x1<x2, y1<y2, rx>0, and ry>0.
        """
        rect = transformNode(self.doc, 'rect', x=x1, y=y1, width=x2 - x1, height=y2 - y1, rx=rx, ry=ry, style=self._formatStyle(AREA_STYLES), **_svgAttrs)
        if link_info:
            rect = self._add_link(rect, link_info)
        self.currGroup.appendChild(rect)

    def drawString(self, s, x, y, angle=0, link_info=None, text_anchor='left', textRenderMode=0, **_svgAttrs):
        if textRenderMode == 3:
            return
        s = asNative(s)
        if self.verbose:
            print('+++ SVGCanvas.drawString')
        needFill = textRenderMode == 0 or textRenderMode == 2 or textRenderMode == 4 or (textRenderMode == 6)
        needStroke = textRenderMode == 1 or textRenderMode == 2 or textRenderMode == 5 or (textRenderMode == 6)
        if self._fillColor != None and needFill or (self._strokeColor != None and needStroke):
            if not text_anchor in ['start', 'inherited', 'left']:
                textLen = stringWidth(s, self._font, self._fontSize)
                if text_anchor == 'end':
                    x -= textLen
                elif text_anchor == 'middle':
                    x -= textLen / 2.0
                elif text_anchor == 'numeric':
                    x -= numericXShift(text_anchor, s, textLen, self._font, self._fontSize)
                else:
                    raise ValueError('bad value for text_anchor ' + str(text_anchor))
            s = self._escape(s)
            st = self._formatStyle(TEXT_STYLES)
            if angle != 0:
                st = st + ' rotate(%s);' % self.fp_str(angle, x, y)
            if needFill:
                st += self._formatStyle(EXTRA_FILL_STYLES)
            else:
                st += ' fill:none;'
            if needStroke:
                st += self._formatStyle(EXTRA_STROKE_STYLES)
            else:
                st += ' stroke:none;'
            text = transformNode(self.doc, 'text', x=x, y=y, style=st, transform='translate(0,%d) scale(1,-1)' % (2 * y), **_svgAttrs)
            content = self.doc.createTextNode(s)
            text.appendChild(content)
            if link_info:
                text = self._add_link(text, link_info)
            self.currGroup.appendChild(text)

    def drawCentredString(self, s, x, y, angle=0, text_anchor='middle', link_info=None, textRenderMode=0, **_svgAttrs):
        if self.verbose:
            print('+++ SVGCanvas.drawCentredString')
        self.drawString(s, x, y, angle=angle, link_info=link_info, text_anchor=text_anchor, textRenderMode=textRenderMode, **_svgAttrs)

    def drawRightString(self, text, x, y, angle=0, text_anchor='end', link_info=None, textRenderMode=0, **_svgAttrs):
        if self.verbose:
            print('+++ SVGCanvas.drawRightString')
        self.drawString(text, x, y, angle=angle, link_info=link_info, text_anchor=text_anchor, textRenderMode=textRenderMode, **_svgAttrs)

    def comment(self, data):
        """Add a comment."""
        comment = self.doc.createComment(data)

    def drawImage(self, image, x, y, width, height, embed=True):
        buf = BytesIO()
        image.save(buf, 'png')
        buf = asNative(base64.b64encode(buf.getvalue()))
        self.currGroup.appendChild(transformNode(self.doc, 'image', x=x, y=y, width=width, height=height, href='data:image/png;base64,' + buf, transform='matrix(%s)' % self.cfp_str(1, 0, 0, -1, 0, height + 2 * y)))

    def line(self, x1, y1, x2, y2):
        if self._strokeColor != None:
            if 0:
                line = transformNode(self.doc, 'line', x=x1, y=y1, x2=x2, y2=y2, style=self._formatStyle(LINE_STYLES))
                self.currGroup.appendChild(line)
            path = transformNode(self.doc, 'path', d='M %s L %s Z' % (self.cfp_str(x1, y1), self.cfp_str(x2, y2)), style=self._formatStyle(LINE_STYLES))
            self.currGroup.appendChild(path)

    def ellipse(self, x1, y1, x2, y2, link_info=None):
        """Draw an orthogonal ellipse inscribed within the rectangle x1,y1,x2,y2.

        These should have x1<x2 and y1<y2.
        """
        ellipse = transformNode(self.doc, 'ellipse', cx=(x1 + x2) / 2.0, cy=(y1 + y2) / 2.0, rx=(x2 - x1) / 2.0, ry=(y2 - y1) / 2.0, style=self._formatStyle(AREA_STYLES))
        if link_info:
            ellipse = self._add_link(ellipse, link_info)
        self.currGroup.appendChild(ellipse)

    def circle(self, xc, yc, r, link_info=None):
        circle = transformNode(self.doc, 'circle', cx=xc, cy=yc, r=r, style=self._formatStyle(AREA_STYLES))
        if link_info:
            circle = self._add_link(circle, link_info)
        self.currGroup.appendChild(circle)

    def drawCurve(self, x1, y1, x2, y2, x3, y3, x4, y4, closed=0):
        pass
        return
        codeline = '%s m %s curveto'
        data = (fp_str(x1, y1), fp_str(x2, y2, x3, y3, x4, y4))
        if self._fillColor != None:
            self.code.append(codeline % data + ' eofill')
        if self._strokeColor != None:
            self.code.append(codeline % data + (closed and ' closepath' or '') + ' stroke')

    def drawArc(self, x1, y1, x2, y2, startAng=0, extent=360, fromcenter=0):
        """Draw a partial ellipse inscribed within the rectangle x1,y1,x2,y2.

        Starting at startAng degrees and covering extent degrees. Angles
        start with 0 to the right (+x) and increase counter-clockwise.
        These should have x1<x2 and y1<y2.
        """
        cx, cy = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        rx, ry = ((x2 - x1) / 2.0, (y2 - y1) / 2.0)
        mx = rx * cos(startAng * pi / 180) + cx
        my = ry * sin(startAng * pi / 180) + cy
        ax = rx * cos((startAng + extent) * pi / 180) + cx
        ay = ry * sin((startAng + extent) * pi / 180) + cy
        cfp_str = self.cfp_str
        s = [].append
        if fromcenter:
            s('M %s L %s' % (cfp_str(cx, cy), cfp_str(ax, ay)))
        if fromcenter:
            s('A %s %d %d %d %s' % (cfp_str(rx, ry), 0, extent >= 180, 0, cfp_str(mx, my)))
        else:
            s('M %s A %s %d %d %d %s Z' % (cfp_str(mx, my), cfp_str(rx, ry), 0, extent >= 180, 0, cfp_str(mx, my)))
        if fromcenter:
            s('L %s Z' % cfp_str(cx, cy))
        path = transformNode(self.doc, 'path', d=' '.join(s.__self__), style=self._formatStyle())
        self.currGroup.appendChild(path)

    def polygon(self, points, closed=0, link_info=None):
        assert len(points) >= 2, 'Polygon must have 2 or more points'
        if self._strokeColor != None or self._fillColor != None:
            pts = ', '.join([fp_str(*p) for p in points])
            polyline = transformNode(self.doc, 'polygon', points=pts, style=self._formatStyle(AREA_STYLES))
            if link_info:
                polyline = self._add_link(polyline, link_info)
            self.currGroup.appendChild(polyline)

    def lines(self, lineList, color=None, width=None):
        return
        if self._strokeColor != None:
            codeline = '%s m %s l stroke'
            for line in lineList:
                self.code.append(codeline % (fp_str(line[0]), fp_str(line[1])))

    def polyLine(self, points):
        assert len(points) >= 1, 'Polyline must have 1 or more points'
        if self._strokeColor != None:
            pts = ', '.join([fp_str(*p) for p in points])
            polyline = transformNode(self.doc, 'polyline', points=pts, style=self._formatStyle(AREA_STYLES, fill=None))
            self.currGroup.appendChild(polyline)

    def startGroup(self, attrDict=dict(transform='')):
        if self.verbose:
            print('+++ begin SVGCanvas.startGroup')
        currGroup = self.currGroup
        group = transformNode(self.doc, 'g', **attrDict)
        currGroup.appendChild(group)
        self.currGroup = group
        if self.verbose:
            print('+++ end SVGCanvas.startGroup')
        return currGroup

    def endGroup(self, currGroup):
        if self.verbose:
            print('+++ begin SVGCanvas.endGroup')
        self.currGroup = currGroup
        if self.verbose:
            print('+++ end SVGCanvas.endGroup')

    def transform(self, a, b, c, d, e, f):
        if self.verbose:
            print('!!! begin SVGCanvas.transform', a, b, c, d, e, f)
        tr = self.currGroup.getAttribute('transform')
        if (a, b, c, d, e, f) != (1, 0, 0, 1, 0, 0):
            t = 'matrix(%s)' % self.cfp_str(a, b, c, d, e, f)
            self.currGroup.setAttribute('transform', '%s %s' % (tr, t))

    def translate(self, x, y):
        if (x, y) != (0, 0):
            self.currGroup.setAttribute('transform', '%s %s' % (self.currGroup.getAttribute('transform'), 'translate(%s)' % self.cfp_str(x, y)))

    def scale(self, sx, sy):
        if (sx, sy) != (1, 1):
            self.currGroup.setAttribute('transform', '%s %s' % (self.groups[-1].getAttribute('transform'), 'scale(%s)' % self.cfp_str(sx, sy)))

    def moveTo(self, x, y):
        self.path = self.path + 'M %s ' % self.fp_str(x, y)

    def lineTo(self, x, y):
        self.path = self.path + 'L %s ' % self.fp_str(x, y)

    def curveTo(self, x1, y1, x2, y2, x3, y3):
        self.path = self.path + 'C %s ' % self.fp_str(x1, y1, x2, y2, x3, y3)

    def closePath(self):
        self.path = self.path + 'Z '

    def saveState(self):
        pass

    def restoreState(self):
        pass