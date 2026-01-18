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
def _listWrapOn(F, availWidth, canv, mergeSpace=1, obj=None, dims=None, fakeWidth=None):
    """return max width, required height for a list of flowables F"""
    doct = getattr(canv, '_doctemplate', None)
    cframe = getattr(doct, 'frame', None)
    if fakeWidth is None:
        fakeWidth = listWrapOnFakeWidth
    if cframe:
        from reportlab.platypus.doctemplate import _addGeneratedContent, Indenter
        doct_frame = cframe
        cframe = doct.frame = deepcopy(doct_frame)
        cframe._generated_content = None
        del cframe._generated_content
    try:
        W = 0
        H = 0
        pS = 0
        atTop = 1
        F = F[:]
        while F:
            f = F.pop(0)
            if hasattr(f, 'frameAction'):
                from reportlab.platypus.doctemplate import Indenter
                if isinstance(f, Indenter):
                    availWidth -= f.left + f.right
                continue
            w, h = f.wrapOn(canv, availWidth, 268435455)
            if dims is not None:
                dims.append((w, h))
            if cframe:
                _addGeneratedContent(F, cframe)
            if w <= _FUZZ and False or h <= _FUZZ:
                continue
            W = max(W, min(w, availWidth) if fakeWidth else w)
            H += h
            if not atTop:
                h = f.getSpaceBefore()
                if mergeSpace:
                    if getattr(f, '_SPACETRANSFER', False):
                        h = pS
                    h = max(h - pS, 0)
                H += h
            else:
                if obj is not None:
                    obj._spaceBefore = f.getSpaceBefore()
                atTop = 0
            s = f.getSpaceAfter()
            if getattr(f, '_SPACETRANSFER', False):
                s = pS
            pS = s
            H += pS
        if obj is not None:
            obj._spaceAfter = pS
        return (W, H - pS)
    finally:
        if cframe:
            doct.frame = doct_frame