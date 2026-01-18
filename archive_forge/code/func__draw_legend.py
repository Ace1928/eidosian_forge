from the canonical composite structure in that we don't really have
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.graphics.shapes import Drawing, String, Line, Rect, Wedge, ArcPath
from reportlab.graphics.widgetbase import Widget
from Bio.Graphics import _write
from Bio.Graphics.GenomeDiagram import _Colors
def _draw_legend(self, cur_drawing, start_y, width):
    """Draw a legend for the figure (PRIVATE).

        Subclasses should implement this (see also self._legend_height) to
        provide specialized legends.
        """