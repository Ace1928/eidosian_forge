from the canonical composite structure in that we don't really have
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.graphics.shapes import Drawing, String, Line, Rect, Wedge, ArcPath
from reportlab.graphics.widgetbase import Widget
from Bio.Graphics import _write
from Bio.Graphics.GenomeDiagram import _Colors
def _draw_segment(self, cur_drawing):
    """Draw a half circle representing the end of a linear chromosome (PRIVATE)."""
    width = (self.end_x_position - self.start_x_position) * self.chr_percent
    height = self.start_y_position - self.end_y_position
    center_x = 0.5 * (self.end_x_position + self.start_x_position)
    start_x = center_x - 0.5 * width
    if self._inverted:
        center_y = self.start_y_position
        start_angle = 180
        end_angle = 360
    else:
        center_y = self.end_y_position
        start_angle = 0
        end_angle = 180
    cap_wedge = Wedge(center_x, center_y, width / 2, start_angle, end_angle, height)
    cap_wedge.strokeColor = None
    cap_wedge.fillColor = self.fill_color
    cur_drawing.add(cap_wedge)
    cap_arc = ArcPath()
    cap_arc.addArc(center_x, center_y, width / 2, start_angle, end_angle, height)
    cur_drawing.add(cap_arc)