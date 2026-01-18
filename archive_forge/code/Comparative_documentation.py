from reportlab.lib import colors
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.graphics.shapes import Drawing, String
from reportlab.graphics.charts.markers import makeEmptySquare, makeFilledSquare
from reportlab.graphics.charts.markers import makeFilledDiamond, makeSmiley
from reportlab.graphics.charts.markers import makeFilledCircle, makeEmptyCircle
from Bio.Graphics import _write
Find min and max for x and y coordinates in the given data (PRIVATE).