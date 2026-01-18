from reportlab.lib.validators import isNumber, isColorOrNone, isNoneOrShape
from reportlab.graphics.widgetbase import Widget
from reportlab.graphics.shapes import Rect, Group, Line, Polygon
from reportlab.lib.attrmap import AttrMap, AttrMapValue
from reportlab.lib.colors import grey
Abstract base class representing a chart's plot area, pretty unusable by itself.