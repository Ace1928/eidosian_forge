from __future__ import annotations
from fontTools.pens.basePen import AbstractPen, DecomposingPen
from fontTools.pens.pointPen import AbstractPointPen, DecomposingPointPen
from fontTools.pens.recordingPen import RecordingPen
class _DecomposingFilterPenMixin:
    """Mixin class that decomposes components as regular contours.

    Shared by both DecomposingFilterPen and DecomposingFilterPointPen.

    Takes two required parameters, another (segment or point) pen 'outPen' to draw
    with, and a 'glyphSet' dict of drawable glyph objects to draw components from.

    The 'skipMissingComponents' and 'reverseFlipped' optional arguments work the
    same as in the DecomposingPen/DecomposingPointPen. Both are False by default.

    In addition, the decomposing filter pens also take the following two options:

    'include' is an optional set of component base glyph names to consider for
    decomposition; the default include=None means decompose all components no matter
    the base glyph name).

    'decomposeNested' (bool) controls whether to recurse decomposition into nested
    components of components (this only matters when 'include' was also provided);
    if False, only decompose top-level components included in the set, but not
    also their children.
    """
    skipMissingComponents = False

    def __init__(self, outPen, glyphSet, skipMissingComponents=None, reverseFlipped=False, include: set[str] | None=None, decomposeNested: bool=True):
        super().__init__(outPen=outPen, glyphSet=glyphSet, skipMissingComponents=skipMissingComponents, reverseFlipped=reverseFlipped)
        self.include = include
        self.decomposeNested = decomposeNested

    def addComponent(self, baseGlyphName, transformation, **kwargs):
        if self.include is None or baseGlyphName in self.include:
            include_bak = self.include
            if self.decomposeNested and self.include:
                self.include = None
            try:
                super().addComponent(baseGlyphName, transformation, **kwargs)
            finally:
                if self.include != include_bak:
                    self.include = include_bak
        else:
            _PassThruComponentsMixin.addComponent(self, baseGlyphName, transformation, **kwargs)