from typing import Tuple, Dict
from fontTools.misc.loggingTools import LogMixin
from fontTools.misc.transform import DecomposedTransform, Identity
class NullPen(AbstractPen):
    """A pen that does nothing."""

    def moveTo(self, pt):
        pass

    def lineTo(self, pt):
        pass

    def curveTo(self, *points):
        pass

    def qCurveTo(self, *points):
        pass

    def closePath(self):
        pass

    def endPath(self):
        pass

    def addComponent(self, glyphName, transformation):
        pass

    def addVarComponent(self, glyphName, transformation, location):
        pass