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
class FrameSplitter(NullDraw):
    """When encountered this flowable should either switch directly to nextTemplate
    if remaining space in the current frame is less than gap+required or it should
    temporarily modify the current template to have the frames from nextTemplate
    that are listed in nextFrames and switch to the first of those frames.
    """
    _ZEROSIZE = 1

    def __init__(self, nextTemplate, nextFrames=[], gap=10, required=72, adjustHeight=True):
        self.nextTemplate = nextTemplate
        self.nextFrames = nextFrames or []
        self.gap = gap
        self.required = required
        self.adjustHeight = adjustHeight

    def wrap(self, aW, aH):
        frame = self._frame
        from reportlab.platypus.doctemplate import NextPageTemplate, CurrentFrameFlowable, LayoutError
        G = [NextPageTemplate(self.nextTemplate)]
        if aH < self.gap + self.required - _FUZZ:
            G.append(PageBreak())
        else:
            templates = self._doctemplateAttr('pageTemplates')
            if templates is None:
                raise LayoutError('%s called in non-doctemplate environment' % self.identity())
            T = [t for t in templates if t.id == self.nextTemplate]
            if not T:
                raise LayoutError('%s.nextTemplate=%s not found' % (self.identity(), self.nextTemplate))
            T = T[0]
            F = [f for f in T.frames if f.id in self.nextFrames]
            N = [f.id for f in F]
            N = [f for f in self.nextFrames if f not in N]
            if N:
                raise LayoutError('%s frames=%r not found in pageTemplate(%s)\n%r has frames %r' % (self.identity(), N, T.id, T, [f.id for f in T.frames]))
            T = self._doctemplateAttr('pageTemplate')

            def unwrap(canv, doc, T=T, onPage=T.onPage, oldFrames=T.frames):
                T.frames = oldFrames
                T.onPage = onPage
                onPage(canv, doc)
            T.onPage = unwrap
            h = aH - self.gap
            for i, f in enumerate(F):
                f = copy(f)
                if self.adjustHeight:
                    f.height = h
                f._reset()
                F[i] = f
            T.frames = F
            G.append(CurrentFrameFlowable(F[0].id))
        frame.add_generated_content(*G)
        return (0, 0)