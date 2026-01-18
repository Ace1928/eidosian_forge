from reportlab.lib import colors
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.graphics.shapes import Drawing, String
from reportlab.graphics.charts.markers import makeEmptySquare, makeFilledSquare
from reportlab.graphics.charts.markers import makeFilledDiamond, makeSmiley
from reportlab.graphics.charts.markers import makeFilledCircle, makeEmptyCircle
from Bio.Graphics import _write
def _find_min_max(self, info):
    """Find min and max for x and y coordinates in the given data (PRIVATE)."""
    x_min = info[0][0][0]
    x_max = info[0][0][0]
    y_min = info[0][0][1]
    y_max = info[0][0][1]
    for two_d_list in info:
        for x, y in two_d_list:
            if x > x_max:
                x_max = x
            if x < x_min:
                x_min = x
            if y > y_max:
                y_max = y
            if y < y_min:
                y_min = y
    return (x_min, x_max, y_min, y_max)