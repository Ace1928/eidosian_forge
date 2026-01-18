from reportlab.graphics.shapes import *
from reportlab.graphics.renderbase import getStateDelta, renderScaledDrawing
from reportlab.pdfbase.pdfmetrics import getFont, unicode2T1
from reportlab.lib.utils import isUnicode
from reportlab import rl_config
from .utils import setFont as _setFont, RenderPMError
import os, sys
from io import BytesIO, StringIO
from math import sin, cos, pi, ceil
from reportlab.graphics.renderbase import Renderer
class PMCanvas:

    def __init__(self, w, h, dpi=72, bg=16777215, configPIL=None, backend=None, backendFmt='RGB'):
        """configPIL dict is passed to image save method"""
        scale = dpi / 72.0
        w = int(w * scale + 0.5)
        h = int(h * scale + 0.5)
        self.__dict__['_gs'] = self._getGState(w, h, bg, backend)
        self.__dict__['_bg'] = bg
        self.__dict__['_baseCTM'] = (scale, 0, 0, scale, 0, 0)
        self.__dict__['_clipPaths'] = []
        self.__dict__['configPIL'] = configPIL
        self.__dict__['_dpi'] = dpi
        self.__dict__['_backend'] = '_renderPM' if type(self._gs._aapixbuf) == type(pow) else 'rlPyCairo'
        self.__dict__['_backendfmt'] = backendFmt
        self.ctm = self._baseCTM

    @staticmethod
    def _getGState(w, h, bg, backend=None, fmt='RGB24'):
        mod = _getPMBackend(backend)
        if backend is None:
            backend = rl_config.renderPMBackend
        if backend == '_renderPM':
            try:
                return mod.gstate(w, h, bg=bg)
            except TypeError:
                try:
                    return mod.GState(w, h, bg, fmt=fmt)
                except:
                    pass
        elif 'cairo' in backend.lower():
            try:
                return mod.GState(w, h, bg, fmt=fmt)
            except AttributeError:
                return mod.gstate(w, h, bg=bg)
        raise RuntimeError(f'Cannot obtain PM graphics state using backend {backend!r}')

    def _drawTimeResize(self, w, h, bg=None):
        if bg is None:
            bg = self._bg
        self._drawing.width, self._drawing.height = (w, h)
        A = {'ctm': None, 'strokeWidth': None, 'strokeColor': None, 'lineCap': None, 'lineJoin': None, 'dashArray': None, 'fillColor': None}
        gs = self._gs
        fN, fS = (gs.fontName, gs.fontSize)
        for k in A.keys():
            A[k] = getattr(gs, k)
        del gs, self._gs
        gs = self.__dict__['_gs'] = _pmBackend.gstate(w, h, bg=bg)
        for k in A.keys():
            setattr(self, k, A[k])
        gs.setFont(fN, fS)

    def toPIL(self):
        im = _getImage().new('RGB', size=(self._gs.width, self._gs.height))
        im.frombytes(self._gs.pixBuf)
        return im

    def saveToFile(self, fn, fmt=None):
        im = self.toPIL()
        if fmt is None:
            if not isinstance(fn, str):
                raise ValueError("Invalid value '%s' for fn when fmt is None" % ascii(fn))
            fmt = os.path.splitext(fn)[1]
            if fmt.startswith('.'):
                fmt = fmt[1:]
        configPIL = self.configPIL or {}
        configPIL.setdefault('preConvertCB', None)
        preConvertCB = configPIL.pop('preConvertCB')
        if preConvertCB:
            im = preConvertCB(im)
        fmt = fmt.upper()
        if fmt in ('GIF',):
            im = _convert2pilp(im)
        elif fmt in ('TIFF', 'TIFFP', 'TIFFL', 'TIF', 'TIFF1'):
            if fmt.endswith('P'):
                im = _convert2pilp(im)
            elif fmt.endswith('L'):
                im = _convert2pilL(im)
            elif fmt.endswith('1'):
                im = _convert2pil1(im)
            fmt = 'TIFF'
        elif fmt in ('PCT', 'PICT'):
            return _saveAsPICT(im, fn, fmt, transparent=configPIL.get('transparent', None))
        elif fmt in ('PNG', 'BMP', 'PPM'):
            pass
        elif fmt in ('JPG', 'JPEG'):
            fmt = 'JPEG'
        else:
            raise RenderPMError('Unknown image kind %s' % fmt)
        if fmt == 'TIFF':
            tc = configPIL.get('transparent', None)
            if tc:
                from PIL import ImageChops, Image
                T = 768 * [0]
                for o, c in zip((0, 256, 512), tc.bitmap_rgb()):
                    T[o + c] = 255
                im = Image.merge('RGBA', im.split() + (ImageChops.invert(im.point(T).convert('L').point(255 * [0] + [255])),))
            for a, d in (('resolution', self._dpi), ('resolution unit', 'inch')):
                configPIL[a] = configPIL.get(a, d)
        configPIL.setdefault('chops_invert', 0)
        if configPIL.pop('chops_invert'):
            from PIL import ImageChops
            im = ImageChops.invert(im)
        configPIL.setdefault('preSaveCB', None)
        preSaveCB = configPIL.pop('preSaveCB')
        if preSaveCB:
            im = preSaveCB(im)
        im.save(fn, fmt, **configPIL)
        if not hasattr(fn, 'write') and os.name == 'mac':
            from reportlab.lib.utils import markfilename
            markfilename(fn, ext=fmt)

    def saveToString(self, fmt='GIF'):
        s = BytesIO()
        self.saveToFile(s, fmt=fmt)
        return s.getvalue()

    def _saveToBMP(self, f):
        """
        Niki Spahiev, <niki@vintech.bg>, asserts that this is a respectable way to get BMP without PIL
        f is a file like object to which the BMP is written
        """
        import struct
        gs = self._gs
        pix, width, height = (gs.pixBuf, gs.width, gs.height)
        f.write(struct.pack('=2sLLLLLLhh24x', 'BM', len(pix) + 54, 0, 54, 40, width, height, 1, 24))
        rowb = width * 3
        for o in range(len(pix), 0, -rowb):
            f.write(pix[o - rowb:o])
        f.write('\x00' * 14)

    def setFont(self, fontName, fontSize, leading=None):
        _setFont(self._gs, fontName, fontSize)

    def __setattr__(self, name, value):
        setattr(self._gs, name, value)

    def __getattr__(self, name):
        return getattr(self._gs, name)

    def fillstrokepath(self, stroke=1, fill=1):
        if fill:
            self.pathFill()
        if stroke:
            self.pathStroke()

    def _bezierArcSegmentCCW(self, cx, cy, rx, ry, theta0, theta1):
        """compute the control points for a bezier arc with theta1-theta0 <= 90.
        Points are computed for an arc with angle theta increasing in the
        counter-clockwise (CCW) direction.  returns a tuple with starting point
        and 3 control points of a cubic bezier curve for the curvto opertator"""
        assert abs(theta1 - theta0) <= 90
        cos0 = cos(pi * theta0 / 180.0)
        sin0 = sin(pi * theta0 / 180.0)
        x0 = cx + rx * cos0
        y0 = cy + ry * sin0
        cos1 = cos(pi * theta1 / 180.0)
        sin1 = sin(pi * theta1 / 180.0)
        x3 = cx + rx * cos1
        y3 = cy + ry * sin1
        dx1 = -rx * sin0
        dy1 = ry * cos0
        halfAng = pi * (theta1 - theta0) / (2.0 * 180.0)
        k = abs(4.0 / 3.0 * (1.0 - cos(halfAng)) / sin(halfAng))
        x1 = x0 + dx1 * k
        y1 = y0 + dy1 * k
        dx2 = -rx * sin1
        dy2 = ry * cos1
        x2 = x3 - dx2 * k
        y2 = y3 - dy2 * k
        return ((x0, y0), ((x1, y1), (x2, y2), (x3, y3)))

    def bezierArcCCW(self, cx, cy, rx, ry, theta0, theta1):
        """return a set of control points for Bezier approximation to an arc
        with angle increasing counter clockwise. No requirement on (theta1-theta0) <= 90
        However, it must be true that theta1-theta0 > 0."""
        angularExtent = theta1 - theta0
        if abs(angularExtent) <= 90.0:
            angleList = [(theta0, theta1)]
        else:
            Nfrag = int(ceil(abs(angularExtent) / 90.0))
            fragAngle = float(angularExtent) / Nfrag
            angleList = []
            for ii in range(Nfrag):
                a = theta0 + ii * fragAngle
                b = a + fragAngle
                angleList.append((a, b))
        ctrlpts = []
        for a, b in angleList:
            if not ctrlpts:
                [(x0, y0), pts] = self._bezierArcSegmentCCW(cx, cy, rx, ry, a, b)
                ctrlpts.append(pts)
            else:
                [(tmpx, tmpy), pts] = self._bezierArcSegmentCCW(cx, cy, rx, ry, a, b)
                ctrlpts.append(pts)
        return ((x0, y0), ctrlpts)

    def addEllipsoidalArc(self, cx, cy, rx, ry, ang1, ang2):
        """adds an ellisesoidal arc segment to a path, with an ellipse centered
        on cx,cy and with radii (major & minor axes) rx and ry.  The arc is
        drawn in the CCW direction.  Requires: (ang2-ang1) > 0"""
        (x0, y0), ctrlpts = self.bezierArcCCW(cx, cy, rx, ry, ang1, ang2)
        self.lineTo(x0, y0)
        for (x1, y1), (x2, y2), (x3, y3) in ctrlpts:
            self.curveTo(x1, y1, x2, y2, x3, y3)

    def drawCentredString(self, x, y, text, text_anchor='middle'):
        self.drawString(x, y, text, text_anchor=text_anchor)

    def drawRightString(self, text, x, y):
        self.drawString(text, x, y, text_anchor='end')

    def drawString(self, x, y, text, _fontInfo=None, text_anchor='left'):
        gs = self._gs
        gs_fontSize = gs.fontSize
        gs_fontName = gs.fontName
        if _fontInfo and _fontInfo != (gs_fontSize, gs_fontName):
            fontName, fontSize = _fontInfo
            _setFont(gs, fontName, fontSize)
        else:
            fontName = gs_fontName
            fontSize = gs_fontSize
        try:
            if text_anchor in ('end', 'middle', 'end'):
                textLen = stringWidth(text, fontName, fontSize)
                if text_anchor == 'end':
                    x -= textLen
                elif text_anchor == 'middle':
                    x -= textLen / 2.0
                elif text_anchor == 'numeric':
                    x -= numericXShift(text_anchor, text, textLen, fontName, fontSize)
            if self._backend == 'rlPyCairo':
                gs.drawString(x, y, text)
            else:
                font = getFont(fontName)
                if font._dynamicFont:
                    gs.drawString(x, y, text)
                else:
                    fc = font
                    if not isUnicode(text):
                        try:
                            text = text.decode('utf8')
                        except UnicodeDecodeError as e:
                            i, j = e.args[2:4]
                            raise UnicodeDecodeError(*e.args[:4] + ('%s\n%s-->%s<--%s' % (e.args[4], text[i - 10:i], text[i:j], text[j:j + 10]),))
                    FT = unicode2T1(text, [font] + font.substitutionFonts)
                    n = len(FT)
                    nm1 = n - 1
                    for i in range(n):
                        f, t = FT[i]
                        if f != fc:
                            _setFont(gs, f.fontName, fontSize)
                            fc = f
                        gs.drawString(x, y, t)
                        if i != nm1:
                            x += f.stringWidth(t.decode(f.encName), fontSize)
        finally:
            gs.setFont(gs_fontName, gs_fontSize)

    def line(self, x1, y1, x2, y2):
        if self.strokeColor is not None:
            self.pathBegin()
            self.moveTo(x1, y1)
            self.lineTo(x2, y2)
            self.pathStroke()

    def rect(self, x, y, width, height, stroke=1, fill=1):
        self.pathBegin()
        self.moveTo(x, y)
        self.lineTo(x + width, y)
        self.lineTo(x + width, y + height)
        self.lineTo(x, y + height)
        self.pathClose()
        self.fillstrokepath(stroke=stroke, fill=fill)

    def roundRect(self, x, y, width, height, rx, ry):
        """rect(self, x, y, width, height, rx,ry):
        Draw a rectangle if rx or rx and ry are specified the corners are
        rounded with ellipsoidal arcs determined by rx and ry
        (drawn in the counter-clockwise direction)"""
        if rx == 0:
            rx = ry
        if ry == 0:
            ry = rx
        x2 = x + width
        y2 = y + height
        self.pathBegin()
        self.moveTo(x + rx, y)
        self.addEllipsoidalArc(x2 - rx, y + ry, rx, ry, 270, 360)
        self.addEllipsoidalArc(x2 - rx, y2 - ry, rx, ry, 0, 90)
        self.addEllipsoidalArc(x + rx, y2 - ry, rx, ry, 90, 180)
        self.addEllipsoidalArc(x + rx, y + ry, rx, ry, 180, 270)
        self.pathClose()
        self.fillstrokepath()

    def circle(self, cx, cy, r):
        """add closed path circle with center cx,cy and axes r: counter-clockwise orientation"""
        self.ellipse(cx, cy, r, r)

    def ellipse(self, cx, cy, rx, ry):
        """add closed path ellipse with center cx,cy and axes rx,ry: counter-clockwise orientation
        (remember y-axis increases downward) """
        self.pathBegin()
        x0 = cx + rx
        y0 = cy
        x3 = cx
        y3 = cy - ry
        x1 = cx + rx
        y1 = cy - ry * BEZIER_ARC_MAGIC
        x2 = x3 + rx * BEZIER_ARC_MAGIC
        y2 = y3
        self.moveTo(x0, y0)
        self.curveTo(x1, y1, x2, y2, x3, y3)
        x0 = x3
        y0 = y3
        x3 = cx - rx
        y3 = cy
        x1 = cx - rx * BEZIER_ARC_MAGIC
        y1 = cy - ry
        x2 = x3
        y2 = cy - ry * BEZIER_ARC_MAGIC
        self.curveTo(x1, y1, x2, y2, x3, y3)
        x0 = x3
        y0 = y3
        x3 = cx
        y3 = cy + ry
        x1 = cx - rx
        y1 = cy + ry * BEZIER_ARC_MAGIC
        x2 = cx - rx * BEZIER_ARC_MAGIC
        y2 = cy + ry
        self.curveTo(x1, y1, x2, y2, x3, y3)
        x0 = x3
        y0 = y3
        x3 = cx + rx
        y3 = cy
        x1 = cx + rx * BEZIER_ARC_MAGIC
        y1 = cy + ry
        x2 = cx + rx
        y2 = cy + ry * BEZIER_ARC_MAGIC
        self.curveTo(x1, y1, x2, y2, x3, y3)
        self.pathClose()

    def saveState(self):
        """do nothing for compatibility"""
        pass

    def setFillColor(self, aColor):
        self.fillColor = Color2Hex(aColor)
        alpha = getattr(aColor, 'alpha', None)
        if alpha is not None:
            self.fillOpacity = alpha

    def setStrokeColor(self, aColor):
        self.strokeColor = Color2Hex(aColor)
        alpha = getattr(aColor, 'alpha', None)
        if alpha is not None:
            self.strokeOpacity = alpha
    restoreState = saveState

    def setLineCap(self, cap):
        self.lineCap = cap

    def setLineJoin(self, join):
        self.lineJoin = join

    def setLineWidth(self, width):
        self.strokeWidth = width