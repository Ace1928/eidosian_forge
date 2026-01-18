from math import log10 as math_log10
from reportlab.lib.validators import    isNumber, isNumberOrNone, isListOfStringsOrNone, isListOfNumbers, \
from reportlab.lib.attrmap import *
from reportlab.lib import normalDate
from reportlab.graphics.shapes import Drawing, Line, PolyLine, Rect, Group, STATE_DEFAULTS, _textBoxLimits, _rotatedBoxLimits
from reportlab.graphics.widgetbase import Widget, TypedPropertyCollection
from reportlab.graphics.charts.textlabels import Label, PMVLabel, XLabel,  DirectDrawFlowable
from reportlab.graphics.charts.utils import nextRoundNumber
from reportlab.graphics.widgets.grids import ShadedRect
from reportlab.lib.colors import Color
from reportlab.lib.utils import isSeq
class AxisLabelAnnotation:
    """Create a grid like line using the given user value to draw the line
    v       value to use
    kwds may contain
    scaleValue  True/not given --> scale the value
                otherwise use the absolute value
    labelClass  the label class to use default Label
    all Label keywords are acceptable (including say _text)
    """

    def __init__(self, v, **kwds):
        self._v = v
        self._kwds = kwds

    def __call__(self, axis):
        kwds = self._kwds.copy()
        labelClass = kwds.pop('labelClass', Label)
        scaleValue = kwds.pop('scaleValue', True)
        if not hasattr(axis, '_tickValues'):
            axis._pseudo_configure()
        sv = (axis.scale if scaleValue else lambda x: x)(self._v)
        if axis.isYAxis:
            x = axis._x
            y = sv
        else:
            x = sv
            y = axis._y
        kwds['x'] = x
        kwds['y'] = y
        return labelClass(**kwds)