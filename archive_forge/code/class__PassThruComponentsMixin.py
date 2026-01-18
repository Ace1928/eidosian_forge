from __future__ import annotations
from fontTools.pens.basePen import AbstractPen, DecomposingPen
from fontTools.pens.pointPen import AbstractPointPen, DecomposingPointPen
from fontTools.pens.recordingPen import RecordingPen
class _PassThruComponentsMixin(object):

    def addComponent(self, glyphName, transformation, **kwargs):
        self._outPen.addComponent(glyphName, transformation, **kwargs)