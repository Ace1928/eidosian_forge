import math
from typing import Any, Optional, Tuple, Dict
from fontTools.misc.loggingTools import LogMixin
from fontTools.pens.basePen import AbstractPen, MissingComponentError, PenError
from fontTools.misc.transform import DecomposedTransform, Identity
class DecomposingPointPen(LogMixin, AbstractPointPen):
    """Implements a 'addComponent' method that decomposes components
    (i.e. draws them onto self as simple contours).
    It can also be used as a mixin class (e.g. see DecomposingRecordingPointPen).

    You must override beginPath, addPoint, endPath. You may
    additionally override addVarComponent and addComponent.

    By default a warning message is logged when a base glyph is missing;
    set the class variable ``skipMissingComponents`` to False if you want
    all instances of a sub-class to raise a :class:`MissingComponentError`
    exception by default.
    """
    skipMissingComponents = True
    MissingComponentError = MissingComponentError

    def __init__(self, glyphSet, *args, skipMissingComponents=None, reverseFlipped=False, **kwargs):
        """Takes a 'glyphSet' argument (dict), in which the glyphs that are referenced
        as components are looked up by their name.

        If the optional 'reverseFlipped' argument is True, components whose transformation
        matrix has a negative determinant will be decomposed with a reversed path direction
        to compensate for the flip.

        The optional 'skipMissingComponents' argument can be set to True/False to
        override the homonymous class attribute for a given pen instance.
        """
        super().__init__(*args, **kwargs)
        self.glyphSet = glyphSet
        self.skipMissingComponents = self.__class__.skipMissingComponents if skipMissingComponents is None else skipMissingComponents
        self.reverseFlipped = reverseFlipped

    def addComponent(self, baseGlyphName, transformation, identifier=None, **kwargs):
        """Transform the points of the base glyph and draw it onto self.

        The `identifier` parameter and any extra kwargs are ignored.
        """
        from fontTools.pens.transformPen import TransformPointPen
        try:
            glyph = self.glyphSet[baseGlyphName]
        except KeyError:
            if not self.skipMissingComponents:
                raise MissingComponentError(baseGlyphName)
            self.log.warning("glyph '%s' is missing from glyphSet; skipped" % baseGlyphName)
        else:
            pen = self
            if transformation != Identity:
                pen = TransformPointPen(pen, transformation)
            if self.reverseFlipped:
                a, b, c, d = transformation[:4]
                det = a * d - b * c
                if a * d - b * c < 0:
                    pen = ReverseContourPointPen(pen)
            glyph.drawPoints(pen)