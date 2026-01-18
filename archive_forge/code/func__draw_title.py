from reportlab.lib import colors
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.graphics.shapes import Drawing, String
from reportlab.graphics.charts.markers import makeEmptySquare, makeFilledSquare
from reportlab.graphics.charts.markers import makeFilledDiamond, makeSmiley
from reportlab.graphics.charts.markers import makeFilledCircle, makeEmptyCircle
from Bio.Graphics import _write
def _draw_title(self, cur_drawing, title, width, height):
    """Add a title to the page we are outputting (PRIVATE)."""
    title_string = String(width / 2, height - inch, title)
    title_string.fontName = 'Helvetica-Bold'
    title_string.fontSize = self.title_size
    title_string.textAnchor = 'middle'
    cur_drawing.add(title_string)