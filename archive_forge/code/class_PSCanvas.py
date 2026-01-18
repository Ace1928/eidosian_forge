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
class PSCanvas:

    def __init__(self, size=(300, 300), PostScriptLevel=2):
        self.width, self.height = size
        xtraState = []
        self._xtraState_push = xtraState.append
        self._xtraState_pop = xtraState.pop
        self.comments = 0
        self.code = []
        self.code_append = self.code.append
        self._sep = '\n'
        self._strokeColor = self._fillColor = self._lineWidth = self._font = self._fontSize = self._lineCap = self._lineJoin = self._color = None
        self._fontsUsed = []
        self.setFont(STATE_DEFAULTS['fontName'], STATE_DEFAULTS['fontSize'])
        self.setStrokeColor(STATE_DEFAULTS['strokeColor'])
        self.setLineCap(2)
        self.setLineJoin(0)
        self.setLineWidth(1)
        self.PostScriptLevel = PostScriptLevel
        self._fillMode = FILL_EVEN_ODD

    def comment(self, msg):
        if self.comments:
            self.code_append('%' + msg)

    def drawImage(self, image, x1, y1, width=None, height=None):
        if self.PostScriptLevel == 1:
            self._drawImageLevel1(image, x1, y1, width, height)
        elif self.PostScriptLevel == 2:
            self._drawImageLevel2(image, x1, y1, width, height)
        else:
            raise ValueError('Unsupported Postscript Level %s' % self.PostScriptLevel)

    def clear(self):
        self.code_append('showpage')

    def _t1_re_encode(self):
        if not self._fontsUsed:
            return
        C = []
        for fontName in self._fontsUsed:
            fontObj = getFont(fontName)
            if not fontObj._dynamicFont and fontObj.encName == 'WinAnsiEncoding':
                C.append('WinAnsiEncoding /%s /%s RE' % (fontName, fontName))
        if C:
            C.insert(0, PS_WinAnsiEncoding)
            self.code.insert(1, self._sep.join(C))

    def save(self, f=None):
        if not hasattr(f, 'write'):
            _f = open(f, 'wb')
        else:
            _f = f
        if self.code[-1] != 'showpage':
            self.clear()
        self.code.insert(0, '%%!PS-Adobe-3.0 EPSF-3.0\n%%%%BoundingBox: 0 0 %d %d\n%%%% Initialization:\n/m {moveto} bind def\n/l {lineto} bind def\n/c {curveto} bind def\n' % (self.width, self.height))
        self._t1_re_encode()
        _f.write(rawBytes(self._sep.join(self.code)))
        if _f is not f:
            _f.close()
            from reportlab.lib.utils import markfilename
            markfilename(f, creatorcode='XPR3', filetype='EPSF')

    def saveState(self):
        self._xtraState_push((self._fontCodeLoc,))
        self.code_append('gsave')

    def restoreState(self):
        self.code_append('grestore')
        self._fontCodeLoc, = self._xtraState_pop()

    def stringWidth(self, s, font=None, fontSize=None):
        """Return the logical width of the string if it were drawn
        in the current font (defaults to self.font)."""
        font = font or self._font
        fontSize = fontSize or self._fontSize
        return stringWidth(s, font, fontSize)

    def setLineCap(self, v):
        if self._lineCap != v:
            self._lineCap = v
            self.code_append('%d setlinecap' % v)

    def setLineJoin(self, v):
        if self._lineJoin != v:
            self._lineJoin = v
            self.code_append('%d setlinejoin' % v)

    def setDash(self, array=[], phase=0):
        """Two notations.  pass two numbers, or an array and phase"""
        psoperation = 'setdash'
        if isinstance(array, (float, int)):
            self.code_append('[%s %s] 0 %s' % (array, phase, psoperation))
        elif isinstance(array, (tuple, list)):
            assert phase >= 0, 'phase is a length in user space'
            textarray = ' '.join(map(str, array))
            self.code_append('[%s] %s %s' % (textarray, phase, psoperation))

    def setStrokeColor(self, color):
        self._strokeColor = color
        self.setColor(color)

    def setColor(self, color):
        if self._color != color:
            self._color = color
            if color:
                if hasattr(color, 'cyan'):
                    self.code_append('%s setcmykcolor' % fp_str(color.cyan, color.magenta, color.yellow, color.black))
                else:
                    self.code_append('%s setrgbcolor' % fp_str(color.red, color.green, color.blue))

    def setFillColor(self, color):
        self._fillColor = color
        self.setColor(color)

    def setFillMode(self, v):
        self._fillMode = v

    def setLineWidth(self, width):
        if width != self._lineWidth:
            self._lineWidth = width
            self.code_append('%s setlinewidth' % width)

    def setFont(self, font, fontSize, leading=None):
        if self._font != font or self._fontSize != fontSize:
            self._fontCodeLoc = len(self.code)
            self._font = font
            self._fontSize = fontSize
            self.code_append('')

    def line(self, x1, y1, x2, y2):
        if self._strokeColor != None:
            self.setColor(self._strokeColor)
            self.code_append('%s m %s l stroke' % (fp_str(x1, y1), fp_str(x2, y2)))

    def _escape(self, s):
        """
        return a copy of string s with special characters in postscript strings
        escaped with backslashes.
        """
        try:
            return _escape_and_limit(s)
        except:
            raise ValueError('cannot escape %s' % ascii(s))

    def _textOut(self, x, y, s, textRenderMode=0):
        if textRenderMode == 3:
            return
        xy = fp_str(x, y)
        s = self._escape(s)
        if textRenderMode == 0:
            self.setColor(self._fillColor)
            self.code_append('%s m (%s) show ' % (xy, s))
            return
        fill = textRenderMode == 0 or textRenderMode == 2 or textRenderMode == 4 or (textRenderMode == 6)
        stroke = textRenderMode == 1 or textRenderMode == 2 or textRenderMode == 5 or (textRenderMode == 6)
        addToClip = textRenderMode >= 4
        if fill and stroke:
            if self._fillColor is None:
                op = ''
            else:
                op = 'fill '
                self.setColor(self._fillColor)
            self.code_append('%s m (%s) true charpath gsave %s' % (xy, s, op))
            self.code_append('grestore ')
            if self._strokeColor is not None:
                self.setColor(self._strokeColor)
                self.code_append('stroke ')
        else:
            self.setColor(self._strokeColor)
            self.code_append('%s m (%s) true charpath stroke ' % (xy, s))

    def _issueT1String(self, fontObj, x, y, s, textRenderMode=0):
        fc = fontObj
        code_append = self.code_append
        fontSize = self._fontSize
        fontsUsed = self._fontsUsed
        escape = self._escape
        if not isUnicode(s):
            try:
                s = s.decode('utf8')
            except UnicodeDecodeError as e:
                i, j = e.args[2:4]
                raise UnicodeDecodeError(*e.args[:4] + ('%s\n%s-->%s<--%s' % (e.args[4], s[i - 10:i], s[i:j], s[j:j + 10]),))
        for f, t in unicode2T1(s, [fontObj] + fontObj.substitutionFonts):
            if f != fc:
                psName = asNative(f.face.name)
                code_append('(%s) findfont %s scalefont setfont' % (psName, fp_str(fontSize)))
                if psName not in fontsUsed:
                    fontsUsed.append(psName)
                fc = f
            self._textOut(x, y, t, textRenderMode)
            x += f.stringWidth(t.decode(f.encName), fontSize)
        if fontObj != fc:
            self._font = None
            self.setFont(fontObj.face.name, fontSize)

    def drawString(self, x, y, s, angle=0, text_anchor='left', textRenderMode=0):
        needFill = textRenderMode in (0, 2, 4, 6)
        needStroke = textRenderMode in (1, 2, 5, 6)
        if needFill or needStroke:
            if text_anchor != 'left':
                textLen = stringWidth(s, self._font, self._fontSize)
                if text_anchor == 'end':
                    x -= textLen
                elif text_anchor == 'middle':
                    x -= textLen / 2.0
                elif text_anchor == 'numeric':
                    x -= numericXShift(text_anchor, s, textLen, self._font, self._fontSize)
            fontObj = getFont(self._font)
            if not self.code[self._fontCodeLoc]:
                psName = asNative(fontObj.face.name)
                self.code[self._fontCodeLoc] = '(%s) findfont %s scalefont setfont' % (psName, fp_str(self._fontSize))
                if psName not in self._fontsUsed:
                    self._fontsUsed.append(psName)
            if angle != 0:
                self.code_append('gsave %s translate %s rotate' % (fp_str(x, y), fp_str(angle)))
                x = y = 0
            oldColor = self._color
            if fontObj._dynamicFont:
                self._textOut(x, y, s, textRenderMode=textRenderMode)
            else:
                self._issueT1String(fontObj, x, y, s, textRenderMode=textRenderMode)
            self.setColor(oldColor)
            if angle != 0:
                self.code_append('grestore')

    def drawCentredString(self, x, y, text, text_anchor='middle', textRenderMode=0):
        self.drawString(x, y, text, text_anchor=text_anchor, textRenderMode=textRenderMode)

    def drawRightString(self, text, x, y, text_anchor='end', textRenderMode=0):
        self.drawString(text, x, y, text_anchor=text_anchor, textRenderMode=textRenderMode)

    def drawCurve(self, x1, y1, x2, y2, x3, y3, x4, y4, closed=0):
        codeline = '%s m %s curveto'
        data = (fp_str(x1, y1), fp_str(x2, y2, x3, y3, x4, y4))
        if self._fillColor != None:
            self.setColor(self._fillColor)
            self.code_append(codeline % data + ' eofill')
        if self._strokeColor != None:
            self.setColor(self._strokeColor)
            self.code_append(codeline % data + (closed and ' closepath' or '') + ' stroke')

    def rect(self, x1, y1, x2, y2, stroke=1, fill=1):
        """Draw a rectangle between x1,y1, and x2,y2"""
        x1, x2 = (min(x1, x2), max(x1, x2))
        y1, y2 = (min(y1, y2), max(y1, y2))
        self.polygon(((x1, y1), (x2, y1), (x2, y2), (x1, y2)), closed=1, stroke=stroke, fill=fill)

    def roundRect(self, x1, y1, x2, y2, rx=8, ry=8):
        """Draw a rounded rectangle between x1,y1, and x2,y2,
        with corners inset as ellipses with x radius rx and y radius ry.
        These should have x1<x2, y1<y2, rx>0, and ry>0."""
        x1, x2 = (min(x1, x2), max(x1, x2))
        y1, y2 = (min(y1, y2), max(y1, y2))
        ellipsePath = 'matrix currentmatrix %s %s translate %s %s scale 0 0 1 %s %s arc setmatrix'
        rr = ['newpath']
        a = rr.append
        a(ellipsePath % (x1 + rx, y1 + ry, rx, -ry, 90, 180))
        a(ellipsePath % (x1 + rx, y2 - ry, rx, -ry, 180, 270))
        a(ellipsePath % (x2 - rx, y2 - ry, rx, -ry, 270, 360))
        a(ellipsePath % (x2 - rx, y1 + ry, rx, -ry, 0, 90))
        a('closepath')
        self._fillAndStroke(rr)

    def ellipse(self, x1, y1, x2, y2):
        """Draw an orthogonal ellipse inscribed within the rectangle x1,y1,x2,y2.
        These should have x1<x2 and y1<y2."""
        self.drawArc(x1, y1, x2, y2)

    def circle(self, xc, yc, r):
        self.ellipse(xc - r, yc - r, xc + r, yc + r)

    def drawArc(self, x1, y1, x2, y2, startAng=0, extent=360, fromcenter=0):
        """Draw a partial ellipse inscribed within the rectangle x1,y1,x2,y2,
        starting at startAng degrees and covering extent degrees.   Angles
        start with 0 to the right (+x) and increase counter-clockwise.
        These should have x1<x2 and y1<y2."""
        cx, cy = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        rx, ry = ((x2 - x1) / 2.0, (y2 - y1) / 2.0)
        codeline = self._genArcCode(x1, y1, x2, y2, startAng, extent)
        startAngleRadians = math.pi * startAng / 180.0
        extentRadians = math.pi * extent / 180.0
        endAngleRadians = startAngleRadians + extentRadians
        codelineAppended = 0
        if self._fillColor != None:
            self.setColor(self._fillColor)
            self.code_append(codeline)
            codelineAppended = 1
            if self._strokeColor != None:
                self.code_append('gsave')
            self.lineTo(cx, cy)
            self.code_append('eofill')
            if self._strokeColor != None:
                self.code_append('grestore')
        if self._strokeColor != None:
            self.setColor(self._strokeColor)
            startx, starty = (cx + rx * math.cos(startAngleRadians), cy + ry * math.sin(startAngleRadians))
            if not codelineAppended:
                self.code_append(codeline)
            if fromcenter:
                self.lineTo(cx, cy)
                self.lineTo(startx, starty)
                self.code_append('closepath')
            self.code_append('stroke')

    def _genArcCode(self, x1, y1, x2, y2, startAng, extent):
        """Calculate the path for an arc inscribed in rectangle defined by (x1,y1),(x2,y2)"""
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

    def polygon(self, p, closed=0, stroke=1, fill=1):
        assert len(p) >= 2, 'Polygon must have 2 or more points'
        start = p[0]
        p = p[1:]
        poly = []
        a = poly.append
        a('%s m' % fp_str(start))
        for point in p:
            a('%s l' % fp_str(point))
        if closed:
            a('closepath')
        self._fillAndStroke(poly, stroke=stroke, fill=fill)

    def lines(self, lineList, color=None, width=None):
        if self._strokeColor != None:
            self._setColor(self._strokeColor)
            codeline = '%s m %s l stroke'
            for line in lineList:
                self.code_append(codeline % (fp_str(line[0]), fp_str(line[1])))

    def moveTo(self, x, y):
        self.code_append('%s m' % fp_str(x, y))

    def lineTo(self, x, y):
        self.code_append('%s l' % fp_str(x, y))

    def curveTo(self, x1, y1, x2, y2, x3, y3):
        self.code_append('%s c' % fp_str(x1, y1, x2, y2, x3, y3))

    def closePath(self):
        self.code_append('closepath')

    def polyLine(self, p):
        assert len(p) >= 1, 'Polyline must have 1 or more points'
        if self._strokeColor != None:
            self.setColor(self._strokeColor)
            self.moveTo(p[0][0], p[0][1])
            for t in p[1:]:
                self.lineTo(t[0], t[1])
            self.code_append('stroke')

    def drawFigure(self, partList, closed=0):
        figureCode = []
        a = figureCode.append
        first = 1
        for part in partList:
            op = part[0]
            args = list(part[1:])
            if op == figureLine:
                if first:
                    first = 0
                    a('%s m' % fp_str(args[:2]))
                else:
                    a('%s l' % fp_str(args[:2]))
                a('%s l' % fp_str(args[2:]))
            elif op == figureArc:
                first = 0
                x1, y1, x2, y2, startAngle, extent = args[:6]
                a(self._genArcCode(x1, y1, x2, y2, startAngle, extent))
            elif op == figureCurve:
                if first:
                    first = 0
                    a('%s m' % fp_str(args[:2]))
                else:
                    a('%s l' % fp_str(args[:2]))
                a('%s curveto' % fp_str(args[2:]))
            else:
                raise TypeError('unknown figure operator: ' + op)
        if closed:
            a('closepath')
        self._fillAndStroke(figureCode)

    def _fillAndStroke(self, code, clip=0, fill=1, stroke=1, fillMode=None):
        fill = self._fillColor and fill
        stroke = self._strokeColor and stroke
        if fill or stroke or clip:
            self.code.extend(code)
            if fill:
                if fillMode is None:
                    fillMode = self._fillMode
                if stroke or clip:
                    self.code_append('gsave')
                self.setColor(self._fillColor)
                self.code_append('eofill' if fillMode == FILL_EVEN_ODD else 'fill')
                if stroke or clip:
                    self.code_append('grestore')
            if stroke:
                if clip:
                    self.code_append('gsave')
                self.setColor(self._strokeColor)
                self.code_append('stroke')
                if clip:
                    self.code_append('grestore')
            if clip:
                self.code_append('clip')
                self.code_append('newpath')

    def translate(self, x, y):
        self.code_append('%s translate' % fp_str(x, y))

    def scale(self, x, y):
        self.code_append('%s scale' % fp_str(x, y))

    def transform(self, a, b, c, d, e, f):
        self.code_append('[%s] concat' % fp_str(a, b, c, d, e, f))

    def _drawTimeResize(self, w, h):
        """if this is used we're probably in the wrong world"""
        self.width, self.height = (w, h)

    def _drawImageLevel1(self, image, x1, y1, width=None, height=None):
        component_depth = 8
        myimage = image.convert('RGB')
        imgwidth, imgheight = myimage.size
        if not width:
            width = imgwidth
        if not height:
            height = imgheight
        self.code.extend(['gsave', '%s %s translate' % (x1, y1), '%s %s scale' % (width, height), '/scanline %d 3 mul string def' % imgwidth])
        self.code.extend(['%s %s %s' % (imgwidth, imgheight, component_depth), '[%s %s %s %s %s %s]' % (imgwidth, 0, 0, -imgheight, 0, imgheight), '{ currentfile scanline readhexstring pop } false 3', 'colorimage '])
        rawimage = (myimage.tobytes if hasattr(myimage, 'tobytes') else myimage.tostring)()
        hex_encoded = self._AsciiHexEncode(rawimage)
        outstream = StringIO(hex_encoded)
        dataline = outstream.read(78)
        while dataline != '':
            self.code_append(dataline)
            dataline = outstream.read(78)
        self.code_append('% end of image data')
        self.code_append('grestore')

    def _AsciiHexEncode(self, input):
        """Helper function used by images"""
        output = StringIO()
        for char in asBytes(input):
            output.write('%02x' % char2int(char))
        return output.getvalue()

    def _drawImageLevel2(self, image, x1, y1, width=None, height=None):
        """At present we're handling only PIL"""
        if image.mode == 'L':
            imBitsPerComponent = 8
            imNumComponents = 1
            myimage = image
        elif image.mode == '1':
            myimage = image.convert('L')
            imNumComponents = 1
            myimage = image
        else:
            myimage = image.convert('RGB')
            imNumComponents = 3
            imBitsPerComponent = 8
        imwidth, imheight = myimage.size
        if not width:
            width = imwidth
        if not height:
            height = imheight
        self.code.extend(['gsave', '%s %s translate' % (x1, y1), '%s %s scale' % (width, height)])
        if imNumComponents == 3:
            self.code_append('/DeviceRGB setcolorspace')
        elif imNumComponents == 1:
            self.code_append('/DeviceGray setcolorspace')
        self.code_append('\n<<\n/ImageType 1\n/Width %d /Height %d  %% dimensions of source image\n/BitsPerComponent %d' % (imwidth, imheight, imBitsPerComponent))
        if imNumComponents == 1:
            self.code_append('/Decode [0 1]')
        if imNumComponents == 3:
            self.code_append('/Decode [0 1 0 1 0 1]  %% decode color values normally')
        self.code.extend(['/ImageMatrix [%s 0 0 %s 0 %s]' % (imwidth, -imheight, imheight), '/DataSource currentfile /ASCIIHexDecode filter', '>> % End image dictionary', 'image'])
        rawimage = (myimage.tobytes if hasattr(myimage, 'tobytes') else myimage.tostring)()
        hex_encoded = self._AsciiHexEncode(rawimage)
        outstream = StringIO(hex_encoded)
        dataline = outstream.read(78)
        while dataline != '':
            self.code_append(dataline)
            dataline = outstream.read(78)
        self.code_append('> % end of image data')
        self.code_append('grestore')