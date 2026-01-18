import math
from typing import Any, Optional, Tuple, Dict
from fontTools.misc.loggingTools import LogMixin
from fontTools.pens.basePen import AbstractPen, MissingComponentError, PenError
from fontTools.misc.transform import DecomposedTransform, Identity
class AbstractPointPen:
    """Baseclass for all PointPens."""

    def beginPath(self, identifier: Optional[str]=None, **kwargs: Any) -> None:
        """Start a new sub path."""
        raise NotImplementedError

    def endPath(self) -> None:
        """End the current sub path."""
        raise NotImplementedError

    def addPoint(self, pt: Tuple[float, float], segmentType: Optional[str]=None, smooth: bool=False, name: Optional[str]=None, identifier: Optional[str]=None, **kwargs: Any) -> None:
        """Add a point to the current sub path."""
        raise NotImplementedError

    def addComponent(self, baseGlyphName: str, transformation: Tuple[float, float, float, float, float, float], identifier: Optional[str]=None, **kwargs: Any) -> None:
        """Add a sub glyph."""
        raise NotImplementedError

    def addVarComponent(self, glyphName: str, transformation: DecomposedTransform, location: Dict[str, float], identifier: Optional[str]=None, **kwargs: Any) -> None:
        """Add a VarComponent sub glyph. The 'transformation' argument
        must be a DecomposedTransform from the fontTools.misc.transform module,
        and the 'location' argument must be a dictionary mapping axis tags
        to their locations.
        """
        raise AttributeError