from __future__ import annotations
from fontTools.pens.basePen import AbstractPen, DecomposingPen
from fontTools.pens.pointPen import AbstractPointPen, DecomposingPointPen
from fontTools.pens.recordingPen import RecordingPen
class DecomposingFilterPointPen(_DecomposingFilterPenMixin, DecomposingPointPen, FilterPointPen):
    """Filter point pen that draws components as regular contours."""
    pass