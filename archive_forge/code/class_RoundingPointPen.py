from fontTools.misc.roundTools import noRound, otRound
from fontTools.misc.transform import Transform
from fontTools.pens.filterPen import FilterPen, FilterPointPen
class RoundingPointPen(FilterPointPen):
    """
    Filter point pen that rounds point coordinates and component XY offsets to integer.
    For rounding the component scale values, a separate round function can be passed to
    the pen.

    >>> from fontTools.pens.recordingPen import RecordingPointPen
    >>> recpen = RecordingPointPen()
    >>> roundpen = RoundingPointPen(recpen)
    >>> roundpen.beginPath()
    >>> roundpen.addPoint((0.4, 0.6), 'line')
    >>> roundpen.addPoint((1.6, 2.5), 'line')
    >>> roundpen.addPoint((2.4, 4.6))
    >>> roundpen.addPoint((3.3, 5.7))
    >>> roundpen.addPoint((4.9, 6.1), 'qcurve')
    >>> roundpen.endPath()
    >>> roundpen.addComponent("a", (1.5, 0, 0, 1.5, 10.5, -10.5))
    >>> recpen.value == [
    ...     ('beginPath', (), {}),
    ...     ('addPoint', ((0, 1), 'line', False, None), {}),
    ...     ('addPoint', ((2, 3), 'line', False, None), {}),
    ...     ('addPoint', ((2, 5), None, False, None), {}),
    ...     ('addPoint', ((3, 6), None, False, None), {}),
    ...     ('addPoint', ((5, 6), 'qcurve', False, None), {}),
    ...     ('endPath', (), {}),
    ...     ('addComponent', ('a', (1.5, 0, 0, 1.5, 11, -10)), {}),
    ... ]
    True
    """

    def __init__(self, outPen, roundFunc=otRound, transformRoundFunc=noRound):
        super().__init__(outPen)
        self.roundFunc = roundFunc
        self.transformRoundFunc = transformRoundFunc

    def addPoint(self, pt, segmentType=None, smooth=False, name=None, identifier=None, **kwargs):
        self._outPen.addPoint((self.roundFunc(pt[0]), self.roundFunc(pt[1])), segmentType=segmentType, smooth=smooth, name=name, identifier=identifier, **kwargs)

    def addComponent(self, baseGlyphName, transformation, identifier=None, **kwargs):
        xx, xy, yx, yy, dx, dy = transformation
        self._outPen.addComponent(baseGlyphName=baseGlyphName, transformation=Transform(self.transformRoundFunc(xx), self.transformRoundFunc(xy), self.transformRoundFunc(yx), self.transformRoundFunc(yy), self.roundFunc(dx), self.roundFunc(dy)), identifier=identifier, **kwargs)