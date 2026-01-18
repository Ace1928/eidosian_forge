from fontTools.pens.basePen import AbstractPen, DecomposingPen
from fontTools.pens.pointPen import AbstractPointPen, DecomposingPointPen
class DecomposingRecordingPen(DecomposingPen, RecordingPen):
    """Same as RecordingPen, except that it doesn't keep components
    as references, but draws them decomposed as regular contours.

    The constructor takes a required 'glyphSet' positional argument,
    a dictionary of glyph objects (i.e. with a 'draw' method) keyed
    by thir name; other arguments are forwarded to the DecomposingPen's
    constructor::

    >>> class SimpleGlyph(object):
    ...     def draw(self, pen):
    ...         pen.moveTo((0, 0))
    ...         pen.curveTo((1, 1), (2, 2), (3, 3))
    ...         pen.closePath()
    >>> class CompositeGlyph(object):
    ...     def draw(self, pen):
    ...         pen.addComponent('a', (1, 0, 0, 1, -1, 1))
    >>> class MissingComponent(object):
    ...     def draw(self, pen):
    ...         pen.addComponent('foobar', (1, 0, 0, 1, 0, 0))
    >>> class FlippedComponent(object):
    ...     def draw(self, pen):
    ...         pen.addComponent('a', (-1, 0, 0, 1, 0, 0))
    >>> glyphSet = {
    ...    'a': SimpleGlyph(),
    ...    'b': CompositeGlyph(),
    ...    'c': MissingComponent(),
    ...    'd': FlippedComponent(),
    ... }
    >>> for name, glyph in sorted(glyphSet.items()):
    ...     pen = DecomposingRecordingPen(glyphSet)
    ...     try:
    ...         glyph.draw(pen)
    ...     except pen.MissingComponentError:
    ...         pass
    ...     print("{}: {}".format(name, pen.value))
    a: [('moveTo', ((0, 0),)), ('curveTo', ((1, 1), (2, 2), (3, 3))), ('closePath', ())]
    b: [('moveTo', ((-1, 1),)), ('curveTo', ((0, 2), (1, 3), (2, 4))), ('closePath', ())]
    c: []
    d: [('moveTo', ((0, 0),)), ('curveTo', ((-1, 1), (-2, 2), (-3, 3))), ('closePath', ())]
    >>> for name, glyph in sorted(glyphSet.items()):
    ...     pen = DecomposingRecordingPen(
    ...         glyphSet, skipMissingComponents=True, reverseFlipped=True,
    ...     )
    ...     glyph.draw(pen)
    ...     print("{}: {}".format(name, pen.value))
    a: [('moveTo', ((0, 0),)), ('curveTo', ((1, 1), (2, 2), (3, 3))), ('closePath', ())]
    b: [('moveTo', ((-1, 1),)), ('curveTo', ((0, 2), (1, 3), (2, 4))), ('closePath', ())]
    c: []
    d: [('moveTo', ((0, 0),)), ('lineTo', ((-3, 3),)), ('curveTo', ((-2, 2), (-1, 1), (0, 0))), ('closePath', ())]
    """
    skipMissingComponents = False