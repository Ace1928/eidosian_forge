from reportlab.lib.colors import _PCMYK_black
from reportlab.graphics.charts.textlabels import Label
from reportlab.graphics.shapes import Circle, Drawing, Group, Line, Rect, String
from reportlab.graphics.widgetbase import Widget
from reportlab.lib.attrmap import *
from reportlab.lib.validators import *
from reportlab.lib.units import cm
from reportlab.pdfbase.pdfmetrics import getFont
from reportlab.graphics.charts.lineplots import _maxWidth
Returns a dotbox widget.