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
def _getContent(self):
    bt = self._bulletType
    value = self._start
    if isinstance(value, (list, tuple)):
        values = value
        value = values[0]
    else:
        values = [value]
    autov = values[0]
    inc = int(bt in '1aAiI')
    if inc:
        try:
            value = int(value)
        except:
            value = 1
    bd = self._bulletDedent
    if bd == 'auto':
        align = self._bulletAlign
        dir = self._bulletDir
        if dir == 'ltr' and align == 'left':
            bd = self._leftIndent
        elif align == 'right':
            bd = self._rightIndent
        else:
            tvalue = value
            maxW = 0
            for d, f in self._flowablesIter():
                if d:
                    maxW = max(maxW, _computeBulletWidth(self, tvalue))
                    if inc:
                        tvalue += inc
                elif isinstance(f, LIIndenter):
                    b = f._bullet
                    if b:
                        if b.bulletType == bt:
                            maxW = max(maxW, _computeBulletWidth(b, b.value))
                            tvalue = int(b.value)
                    else:
                        maxW = max(maxW, _computeBulletWidth(self, tvalue))
                    if inc:
                        tvalue += inc
            if dir == 'ltr':
                if align == 'right':
                    bd = self._leftIndent - maxW
                else:
                    bd = self._leftIndent - maxW * 0.5
            elif align == 'left':
                bd = self._rightIndent - maxW
            else:
                bd = self._rightIndent - maxW * 0.5
    self._calcBulletDedent = bd
    S = []
    aS = S.append
    i = 0
    for d, f in self._flowablesIter():
        if isinstance(f, ListFlowable):
            fstart = f._start
            if isinstance(fstart, (list, tuple)):
                fstart = fstart[0]
            if fstart in values:
                if f._auto:
                    autov = values.index(autov) + 1
                    f._start = values[autov:] + values[:autov]
                    autov = f._start[0]
                    if inc:
                        f._bulletType = autov
                else:
                    autov = fstart
        fparams = {}
        if not i:
            i += 1
            spaceBefore = getattr(self, 'spaceBefore', None)
            if spaceBefore is not None:
                fparams['spaceBefore'] = spaceBefore
        if d:
            aS(self._makeLIIndenter(f, bullet=self._makeBullet(value), params=fparams))
            if inc:
                value += inc
        elif isinstance(f, LIIndenter):
            b = f._bullet
            if b:
                if b.bulletType != bt:
                    raise ValueError('Included LIIndenter bulletType=%s != OrderedList bulletType=%s' % (b.bulletType, bt))
                value = int(b.value)
            else:
                f._bullet = self._makeBullet(value, params=getattr(f, 'params', None))
            if fparams:
                f.__dict__['spaceBefore'] = max(f.__dict__.get('spaceBefore', 0), spaceBefore)
            aS(f)
            if inc:
                value += inc
        elif isinstance(f, _LIParams):
            fparams.update(f.params)
            z = self._makeLIIndenter(f.flowable, bullet=None, params=fparams)
            if f.first:
                if f.value is not None:
                    value = f.value
                    if inc:
                        value = int(value)
                z._bullet = self._makeBullet(value, f.params)
                if inc:
                    value += inc
            aS(z)
        else:
            aS(self._makeLIIndenter(f, bullet=None, params=fparams))
    spaceAfter = getattr(self, 'spaceAfter', None)
    if spaceAfter is not None:
        f = S[-1]
        f.__dict__['spaceAfter'] = max(f.__dict__.get('spaceAfter', 0), spaceAfter)
    if self._caption:
        S.insert(0, self._caption)
    return S