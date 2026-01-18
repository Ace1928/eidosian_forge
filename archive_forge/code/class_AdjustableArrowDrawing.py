from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from reportlab.graphics.shapes import Drawing, _DrawingEditorMixin, Group, Polygon
from reportlab.graphics.widgetbase import Widget
class AdjustableArrowDrawing(_DrawingEditorMixin, Drawing):

    def __init__(self, width=100, height=63, *args, **kw):
        Drawing.__init__(self, width, height, *args, **kw)
        self._add(self, AdjustableArrow(), name='adjustableArrow', validate=None, desc=None)