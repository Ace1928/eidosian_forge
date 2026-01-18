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
def _findSplit(self, canv, availWidth, availHeight, mergeSpace=1, obj=None, content=None, paraFix=True):
    """return max width, required height for a list of flowables F"""
    W = 0
    H = 0
    pS = sB = 0
    atTop = 1
    F = self._getContent(content)
    for i, f in enumerate(F):
        if hasattr(f, 'frameAction'):
            from reportlab.platypus.doctemplate import Indenter
            if isinstance(f, Indenter):
                availWidth -= f.left + f.right
            continue
        w, h = f.wrapOn(canv, availWidth, 268435455)
        if w <= _FUZZ or h <= _FUZZ:
            continue
        W = max(W, w)
        if not atTop:
            s = f.getSpaceBefore()
            if mergeSpace:
                s = max(s - pS, 0)
            H += s
        else:
            if obj is not None:
                obj._spaceBefore = f.getSpaceBefore()
            atTop = 0
        if H >= availHeight or w > availWidth:
            return (W, availHeight, F[:i], F[i:])
        H += h
        if H > availHeight:
            aH = availHeight - (H - h)
            if paraFix:
                from reportlab.platypus.paragraph import Paragraph
                if isinstance(f, (Paragraph, Preformatted)):
                    leading = f.style.leading
                    nH = leading * int(aH / float(leading)) + _FUZZ
                    if nH < aH:
                        nH += leading
                    availHeight += nH - aH
                    aH = nH
            try:
                S = cdeepcopy(f).splitOn(canv, availWidth, aH)
            except:
                S = None
            if not S:
                return (W, availHeight, F[:i], F[i:])
            else:
                return (W, availHeight, F[:i] + S[:1], S[1:] + F[i + 1:])
        pS = f.getSpaceAfter()
        H += pS
    if obj is not None:
        obj._spaceAfter = pS
    return (W, H - pS, F, [])