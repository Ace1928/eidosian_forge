import math
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing, String
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.barcharts import BarChartProperties
from reportlab.graphics.widgetbase import TypedPropertyCollection
from Bio.Graphics import _write
Draw a line distribution into the current drawing.