from reportlab.graphics import shapes
from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from weakref import ref as weakref_ref
class StyleProperties(PropHolder):
    """A container class for attributes used in charts and legends.

    Attributes contained can be those for any graphical element
    (shape?) in the ReportLab graphics package. The idea for this
    container class is to be useful in combination with legends
    and/or the individual appearance of data series in charts.

    A legend could be as simple as a wrapper around a list of style
    properties, where the 'desc' attribute contains a descriptive
    string and the rest could be used by the legend e.g. to draw
    something like a color swatch. The graphical presentation of
    the legend would be its own business, though.

    A chart could be inspecting a legend or, more directly, a list
    of style properties to pick individual attributes that it knows
    about in order to render a particular row of the data. A bar
    chart e.g. could simply use 'strokeColor' and 'fillColor' for
    drawing the bars while a line chart could also use additional
    ones like strokeWidth.
    """
    _attrMap = AttrMap(strokeWidth=AttrMapValue(isNumber, desc='width of the stroke line'), strokeLineCap=AttrMapValue(isNumber, desc='Line cap 0=butt, 1=round & 2=square', advancedUsage=1), strokeLineJoin=AttrMapValue(isNumber, desc='Line join 0=miter, 1=round & 2=bevel', advancedUsage=1), strokeMiterLimit=AttrMapValue(None, desc='miter limit control miter line joins', advancedUsage=1), strokeDashArray=AttrMapValue(isListOfNumbersOrNone, desc='dashing patterns e.g. (1,3)'), strokeOpacity=AttrMapValue(isNumber, desc='level of transparency (alpha) accepts values between 0..1', advancedUsage=1), strokeColor=AttrMapValue(isColorOrNone, desc='the color of the stroke'), fillColor=AttrMapValue(isColorOrNone, desc='the filling color'), desc=AttrMapValue(isString))

    def __init__(self, **kwargs):
        """Initialize with attributes if any."""
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __setattr__(self, name, value):
        """Verify attribute name and value, before setting it."""
        validateSetattr(self, name, value)