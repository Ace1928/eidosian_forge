import os
from copy import deepcopy, copy
from reportlab.lib.colors import gray, lightgrey
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.lib.styles import _baseFontName
from reportlab.lib.utils import strTypes, rl_safe_exec, annotateException
from reportlab.lib.abag import ABag
from reportlab.pdfbase import pdfutils
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.rl_config import _FUZZ, overlapAttachedSpace, ignoreContainerActions, listWrapOnFakeWidth
from reportlab.lib.sequencer import _type2formatter
from reportlab.lib.styles import ListStyle
class BalancedColumns(_FindSplitterMixin, NullDraw):
    """combine a list of flowables and an Image"""

    def __init__(self, F, nCols=2, needed=72, spaceBefore=0, spaceAfter=0, showBoundary=None, leftPadding=None, innerPadding=None, rightPadding=None, topPadding=None, bottomPadding=None, name='', endSlack=0.1, boxStrokeColor=None, boxStrokeWidth=0, boxFillColor=None, boxMargin=None, vLinesStrokeColor=None, vLinesStrokeWidth=None):
        self.name = name or 'BalancedColumns-%d' % id(self)
        if nCols < 2:
            raise ValueError('nCols should be at least 2 not %r in %s' % (nCols, self.identitity()))
        self._content = _flowableSublist(F)
        self._nCols = nCols
        self.spaceAfter = spaceAfter
        self._leftPadding = leftPadding
        self._innerPadding = innerPadding
        self._rightPadding = rightPadding
        self._topPadding = topPadding
        self._bottomPadding = bottomPadding
        self.spaceBefore = spaceBefore
        self._needed = needed - _FUZZ
        self.showBoundary = showBoundary
        self.endSlack = endSlack
        self._boxStrokeColor = boxStrokeColor
        self._boxStrokeWidth = boxStrokeWidth
        self._boxFillColor = boxFillColor
        self._boxMargin = boxMargin
        self._vLinesStrokeColor = vLinesStrokeColor
        self._vLinesStrokeWidth = vLinesStrokeWidth

    def identity(self, maxLen=None):
        return '<%s nCols=%r at %s%s%s>' % (self.__class__.__name__, self._nCols, hex(id(self)), self._frameName(), getattr(self, 'name', '') and ' name="%s"' % getattr(self, 'name', '') or '')

    def getSpaceAfter(self):
        return self.spaceAfter

    def getSpaceBefore(self):
        return self.spaceBefore

    def _generated_content(self, aW, aH):
        G = []
        frame = self._frame
        from reportlab.platypus.doctemplate import LayoutError, ActionFlowable, Indenter
        from reportlab.platypus.frames import Frame
        from reportlab.platypus.doctemplate import FrameBreak
        lpad = frame._leftPadding if self._leftPadding is None else self._leftPadding
        rpad = frame._rightPadding if self._rightPadding is None else self._rightPadding
        tpad = frame._topPadding if self._topPadding is None else self._topPadding
        bpad = frame._bottomPadding if self._bottomPadding is None else self._bottomPadding
        leftExtraIndent = frame._leftExtraIndent
        rightExtraIndent = frame._rightExtraIndent
        gap = max(lpad, rpad) if self._innerPadding is None else self._innerPadding
        hgap = gap * 0.5
        canv = self.canv
        nCols = self._nCols
        cw = (aW - gap * (nCols - 1) - lpad - rpad) / float(nCols)
        aH0 = aH
        aH -= tpad + bpad
        W, H0, _C0, C2 = self._findSplit(canv, cw, nCols * aH, paraFix=False)
        if not _C0:
            raise ValueError('%s cannot make initial split aW=%r aH=%r ie cw=%r ah=%r\ncontent=%s' % (self.identity(), aW, aH, cw, nCols * aH, [f.__class__.__name__ for f in self._content]))
        _fres = {}

        def splitFunc(ah, endSlack=0):
            if ah not in _fres:
                c = []
                w = 0
                h = 0
                cn = None
                icheck = nCols - 2 if endSlack else -1
                for i in range(nCols):
                    wi, hi, c0, c1 = self._findSplit(canv, cw, ah, content=cn, paraFix=False)
                    w = max(w, wi)
                    h = max(h, hi)
                    c.append(c0)
                    if i == icheck:
                        wc, hc, cc0, cc1 = self._findSplit(canv, cw, 2 * ah, content=c1, paraFix=False)
                        if hc <= (1 + endSlack) * ah:
                            c.append(c1)
                            h = ah - 1e-06
                            cn = []
                            break
                    cn = c1
                _fres[ah] = (ah + 100000 * int(cn != []), cn == [], (w, h, c, cn))
            return _fres[ah][2]
        endSlack = 0
        if C2:
            H = aH
        else:
            import math

            def func(ah):
                splitFunc(ah)
                return _fres[ah][0]

            def gss(f, a, b, tol=1, gr=(math.sqrt(5) + 1) / 2):
                c = b - (b - a) / gr
                d = a + (b - a) / gr
                while abs(a - b) > tol:
                    if f(c) < f(d):
                        b = d
                    else:
                        a = c
                    c = b - (b - a) / gr
                    d = a + (b - a) / gr
                F = [(x, tf, v) for x, tf, v in _fres.values() if tf]
                if F:
                    F.sort()
                    return F[0][2]
                return None
            H = min(int(H0 / float(nCols) + self.spaceAfter * 0.4), aH)
            splitFunc(H)
            if not _fres[H][1]:
                H = gss(func, H, aH)
                if H:
                    W, H0, _C0, C2 = H
                    H = H0
                    endSlack = False
                else:
                    H = aH
                    endSlack = self.endSlack
            else:
                H1 = H0 / float(nCols)
                splitFunc(H1)
                if not _fres[H1][1]:
                    H = gss(func, H, aH)
                    if H:
                        W, H0, _C0, C2 = H
                        H = H0
                        endSlack = False
                    else:
                        H = aH
                        endSlack = self.endSlack
            assert not C2, 'unexpected non-empty C2'
        W1, H1, C, C1 = splitFunc(H, endSlack)
        _fres.clear()
        if C[0] == [] and C[1] == [] and C1:
            C, C1 = ([C1, C[1]], C[0])
        x1 = frame._x1
        y1 = frame._y1
        fw = frame._width
        ftop = y1 + bpad + tpad + aH
        fh = H1 + bpad + tpad
        y2 = ftop - fh
        dx = aW / float(nCols)
        if leftExtraIndent or rightExtraIndent:
            indenter0 = Indenter(-leftExtraIndent, -rightExtraIndent)
            indenter1 = Indenter(leftExtraIndent, rightExtraIndent)
        else:
            indenter0 = indenter1 = None
        showBoundary = self.showBoundary if self.showBoundary is not None else frame.showBoundary
        obx = x1 + leftExtraIndent + frame._leftPadding
        F = [Frame(obx + i * dx, y2, dx, fh, leftPadding=lpad if not i else hgap, bottomPadding=bpad, rightPadding=rpad if i == nCols - 1 else hgap, topPadding=tpad, id='%s-%d' % (self.name, i), showBoundary=showBoundary, overlapAttachedSpace=frame._oASpace, _debug=frame._debug) for i in range(nCols)]
        T = self._doctemplateAttr('pageTemplate')
        if T is None:
            raise LayoutError('%s used in non-doctemplate environment' % self.identity())
        BGs = getattr(frame, '_frameBGs', None)
        xbg = bg = BGs[-1] if BGs else None

        class TAction(ActionFlowable):
            """a special Action flowable that sets stuff on the doc template T"""

            def __init__(self, bgs=[], F=[], f=None):
                Flowable.__init__(self)
                self.bgs = bgs
                self.F = F
                self.f = f

            def apply(self, doc, T=T):
                T.frames = self.F
                frame._frameBGs = self.bgs
                doc.handle_currentFrame(self.f.id)
                frame._frameBGs = self.bgs
        if bg:
            xbg = _ExtendBG(y2, fh, bg, frame)
            G.append(xbg)
        oldFrames = T.frames
        G.append(TAction([], F, F[0]))
        if indenter0:
            G.append(indenter0)
        doBox = self._boxStrokeColor and self._boxStrokeWidth and (self._boxStrokeWidth >= 0) or self._boxFillColor
        doVLines = self._vLinesStrokeColor and self._vLinesStrokeWidth and (self._vLinesStrokeWidth >= 0)
        if doBox or doVLines:
            obm = self._boxMargin
            if not obm:
                obm = (0, 0, 0, 0)
            if len(obm) == 1:
                obmt = obml = obmr = obmb = obm[0]
            elif len(obm) == 2:
                obmt = obmb = obm[0]
                obml = obmr = obm[1]
            elif len(obm) == 3:
                obmt = obm[0]
                obml = obmr = obm[1]
                obmb = obm[2]
            elif len(obm) == 4:
                obmt = obm[0]
                obmr = obm[1]
                obmb = obm[2]
                obml = obm[3]
            else:
                raise ValueError('Invalid value %s for boxMargin' % repr(obm))
            obx1 = obx - obml
            obx2 = F[-1]._x1 + F[-1]._width + obmr
            oby2 = y2 - obmb
            obh = fh + obmt + obmb
            oby1 = oby2 + obh
            if doBox:
                box = _AbsRect(obx1, oby2, obx2 - obx1, obh, fillColor=self._boxFillColor, strokeColor=self._boxStrokeColor, strokeWidth=self._boxStrokeWidth)
            if doVLines:
                vLines = []
                for i in range(1, nCols):
                    vlx = 0.5 * (F[i]._x1 + F[i - 1]._x1 + F[i - 1]._width)
                    vLines.append(_AbsLine(vlx, oby2, vlx, oby1, strokeWidth=self._vLinesStrokeWidth, strokeColor=self._vLinesStrokeColor))
        else:
            oby1 = ftop
            oby2 = y2
        if doBox:
            G.append(box)
        if doVLines:
            G.extend(vLines)
        sa = self.getSpaceAfter()
        for i in range(nCols):
            Ci = C[i]
            if Ci:
                Ci = KeepInFrame(W1, H1, Ci, mode='shrink')
                sa = max(sa, Ci.getSpaceAfter())
                G.append(Ci)
            if i != nCols - 1:
                G.append(FrameBreak)
        G.append(TAction(BGs, oldFrames, frame))
        if xbg:
            if C1:
                sa = 0
            xbg._y = min(y2, oby2) - sa
            xbg._height = max(ftop, oby1) - xbg._y
        if indenter1:
            G.append(indenter1)
        if C1:
            G.append(BalancedColumns(C1, nCols=nCols, needed=self._needed, spaceBefore=self.spaceBefore, spaceAfter=self.spaceAfter, showBoundary=self.showBoundary, leftPadding=self._leftPadding, innerPadding=self._innerPadding, rightPadding=self._rightPadding, topPadding=self._topPadding, bottomPadding=self._bottomPadding, name=self.name + '-1', endSlack=self.endSlack, boxStrokeColor=self._boxStrokeColor, boxStrokeWidth=self._boxStrokeWidth, boxFillColor=self._boxFillColor, boxMargin=self._boxMargin, vLinesStrokeColor=self._vLinesStrokeColor, vLinesStrokeWidth=self._vLinesStrokeWidth))
        return (fh, G)

    def wrap(self, aW, aH):
        self_frame = getattr(self, '_frame', None)
        if aH < self.spaceBefore + self._needed - _FUZZ:
            G = [PageBreak(), self]
            H1 = 0
        else:
            if not self_frame:
                from reportlab.platypus.frames import Frame
                self._frame = Frame(0, 0, aW, 2147483647, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
            H1, G = self._generated_content(aW, aH)
            if not self_frame:
                del self._frame
        if self_frame:
            self_frame.add_generated_content(*G)
        return (0, min(H1, aH))