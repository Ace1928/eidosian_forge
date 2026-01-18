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
class KeepInFrame(_Container, Flowable):

    def __init__(self, maxWidth, maxHeight, content=[], mergeSpace=1, mode='shrink', name='', hAlign='LEFT', vAlign='BOTTOM', fakeWidth=None):
        """mode describes the action to take when overflowing
            error       raise an error in the normal way
            continue    ignore ie just draw it and report maxWidth, maxHeight
            shrink      shrinkToFit
            truncate    fit as much as possible
            set fakeWidth to False to make _listWrapOn do the 'right' thing
        """
        self.name = name
        self.maxWidth = maxWidth
        self.maxHeight = maxHeight
        self.mode = mode
        assert mode in ('error', 'overflow', 'shrink', 'truncate'), '%s invalid mode value %s' % (self.identity(), mode)
        assert maxHeight >= 0, '%s invalid maxHeight value %s' % (self.identity(), maxHeight)
        if mergeSpace is None:
            mergeSpace = overlapAttachedSpace
        self.mergespace = mergeSpace
        self._content = content or []
        self.vAlign = vAlign
        self.hAlign = hAlign
        self.fakeWidth = fakeWidth

    def _getAvailableWidth(self):
        return self.maxWidth - self._leftExtraIndent - self._rightExtraIndent

    def identity(self, maxLen=None):
        return '<%s at %s%s%s> size=%sx%s' % (self.__class__.__name__, hex(id(self)), self._frameName(), getattr(self, 'name', '') and ' name="%s"' % getattr(self, 'name', '') or '', getattr(self, 'maxWidth', '') and ' maxWidth=%s' % fp_str(getattr(self, 'maxWidth', 0)) or '', getattr(self, 'maxHeight', '') and ' maxHeight=%s' % fp_str(getattr(self, 'maxHeight')) or '')

    def wrap(self, availWidth, availHeight):
        from reportlab.platypus.doctemplate import LayoutError
        mode = self.mode
        maxWidth = float(min(self.maxWidth or availWidth, availWidth))
        maxHeight = float(min(self.maxHeight or availHeight, availHeight))
        fakeWidth = self.fakeWidth
        W, H = _listWrapOn(self._content, maxWidth, self.canv, fakeWidth=fakeWidth)
        if mode == 'error' and (W > maxWidth + _FUZZ or H > maxHeight + _FUZZ):
            ident = 'content %sx%s too large for %s' % (W, H, self.identity(30))
            raise LayoutError(ident)
        elif W <= maxWidth + _FUZZ and H <= maxHeight + _FUZZ:
            self.width = W - _FUZZ
            self.height = H - _FUZZ
        elif mode in ('overflow', 'truncate'):
            self.width = min(maxWidth, W) - _FUZZ
            self.height = min(maxHeight, H) - _FUZZ
        else:

            def func(x):
                x = float(x)
                W, H = _listWrapOn(self._content, x * maxWidth, self.canv, fakeWidth=fakeWidth)
                W /= x
                H /= x
                return (W, H)
            W0 = W
            H0 = H
            s0 = 1
            if W > maxWidth + _FUZZ:
                s1 = W / maxWidth
                W, H = func(s1)
                if H <= maxHeight + _FUZZ:
                    self.width = W - _FUZZ
                    self.height = H - _FUZZ
                    self._scale = s1
                    return (W, H)
                s0 = s1
                H0 = H
                W0 = W
            s1 = H / maxHeight
            W, H = func(s1)
            self.width = W - _FUZZ
            self.height = H - _FUZZ
            self._scale = s1
            if H < min(0.95 * maxHeight, maxHeight - 10) or H >= maxHeight + _FUZZ:
                H1 = H
                for f in (0, 0.01, 0.05, 0.1, 0.15):
                    s = _qsolve(maxHeight * (1 - f), _hmodel(s0, s1, H0, H1))
                    W, H = func(s)
                    if H <= maxHeight + _FUZZ and W <= maxWidth + _FUZZ:
                        self.width = W - _FUZZ
                        self.height = H - _FUZZ
                        self._scale = s
                        break
        return (self.width, self.height)

    def drawOn(self, canv, x, y, _sW=0):
        scale = getattr(self, '_scale', 1.0)
        truncate = self.mode == 'truncate'
        ss = scale != 1.0 or truncate
        if ss:
            canv.saveState()
            if truncate:
                p = canv.beginPath()
                p.rect(x, y, self.width, self.height)
                canv.clipPath(p, stroke=0)
            else:
                canv.translate(x, y)
                x = y = 0
                canv.scale(1.0 / scale, 1.0 / scale)
        _Container.drawOn(self, canv, x, y, _sW=_sW, scale=scale)
        if ss:
            canv.restoreState()