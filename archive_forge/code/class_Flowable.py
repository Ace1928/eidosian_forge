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
class Flowable:
    """Abstract base class for things to be drawn.  Key concepts:

    1. It knows its size
    2. It draws in its own coordinate system (this requires the
       base API to provide a translate() function.

    """
    _fixedWidth = 0
    _fixedHeight = 0

    def __init__(self):
        self.width = 0
        self.height = 0
        self.wrapped = 0
        self.hAlign = 'LEFT'
        self.vAlign = 'BOTTOM'
        self._traceInfo = None
        self._showBoundary = None
        self.encoding = None

    def _drawOn(self, canv):
        """ensure canv is set on and then draw"""
        self.canv = canv
        self.draw()
        del self.canv

    def _hAlignAdjust(self, x, sW=0):
        if sW and hasattr(self, 'hAlign'):
            a = self.hAlign
            if a in ('CENTER', 'CENTRE', TA_CENTER):
                x += 0.5 * sW
            elif a in ('RIGHT', TA_RIGHT):
                x += sW
            elif a not in ('LEFT', TA_LEFT):
                raise ValueError('Bad hAlign value ' + str(a))
        return x

    def drawOn(self, canvas, x, y, _sW=0):
        """Tell it to draw itself on the canvas.  Do not override"""
        x = self._hAlignAdjust(x, _sW)
        canvas.saveState()
        canvas.translate(x, y)
        self._drawOn(canvas)
        if hasattr(self, '_showBoundary') and self._showBoundary:
            canvas.setStrokeColor(gray)
            canvas.rect(0, 0, self.width, self.height)
        canvas.restoreState()

    def wrapOn(self, canv, aW, aH):
        """intended for use by packers allows setting the canvas on
        during the actual wrap"""
        self.canv = canv
        w, h = self.wrap(aW, aH)
        del self.canv
        return (w, h)

    def wrap(self, availWidth, availHeight):
        """This will be called by the enclosing frame before objects
        are asked their size, drawn or whatever.  It returns the
        size actually used."""
        return (self.width, self.height)

    def minWidth(self):
        """This should return the minimum required width"""
        return getattr(self, '_minWidth', self.width)

    def splitOn(self, canv, aW, aH):
        """intended for use by packers allows setting the canvas on
        during the actual split"""
        self.canv = canv
        S = self.split(aW, aH)
        del self.canv
        return S

    def split(self, availWidth, availheight):
        """This will be called by more sophisticated frames when
        wrap fails. Stupid flowables should return []. Clever flowables
        should split themselves and return a list of flowables.
        If they decide that nothing useful can be fitted in the
        available space (e.g. if you have a table and not enough
        space for the first row), also return []"""
        return []

    def getKeepWithNext(self):
        """returns boolean determining whether the next flowable should stay with this one"""
        if hasattr(self, 'keepWithNext'):
            return self.keepWithNext
        elif hasattr(self, 'style') and hasattr(self.style, 'keepWithNext'):
            return self.style.keepWithNext
        else:
            return 0

    def getSpaceAfter(self):
        """returns how much space should follow this item if another item follows on the same page."""
        if hasattr(self, 'spaceAfter'):
            return self.spaceAfter
        elif hasattr(self, 'style') and hasattr(self.style, 'spaceAfter'):
            return self.style.spaceAfter
        else:
            return 0

    def getSpaceBefore(self):
        """returns how much space should precede this item if another item precedess on the same page."""
        if hasattr(self, 'spaceBefore'):
            return self.spaceBefore
        elif hasattr(self, 'style') and hasattr(self.style, 'spaceBefore'):
            return self.style.spaceBefore
        else:
            return 0

    def isIndexing(self):
        """Hook for IndexingFlowables - things which have cross references"""
        return 0

    def identity(self, maxLen=None):
        """
        This method should attempt to return a string that can be used to identify
        a particular flowable uniquely. The result can then be used for debugging
        and or error printouts
        """
        if hasattr(self, 'getPlainText'):
            r = self.getPlainText(identify=1)
        elif hasattr(self, 'text'):
            r = str(self.text)
        else:
            r = '...'
        if r and maxLen:
            r = r[:maxLen]
        return '<%s at %s%s>%s' % (self.__class__.__name__, hex(id(self)), self._frameName(), r)

    @property
    def _doctemplate(self):
        return getattr(getattr(self, 'canv', None), '_doctemplate', None)

    def _doctemplateAttr(self, a):
        return getattr(self._doctemplate, a, None)

    def _frameName(self):
        f = getattr(self, '_frame', None)
        if not f:
            f = self._doctemplateAttr('frame')
        if f and f.id:
            return ' frame=%s' % f.id
        return ''