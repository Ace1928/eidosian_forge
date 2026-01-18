from the canonical composite structure in that we don't really have
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.graphics.shapes import Drawing, String, Line, Rect, Wedge, ArcPath
from reportlab.graphics.widgetbase import Widget
from Bio.Graphics import _write
from Bio.Graphics.GenomeDiagram import _Colors
def _draw_label(self, cur_drawing):
    """Add a label to the chromosome segment (PRIVATE).

        The label will be applied to the right of the segment.

        This may be overlapped by any sub-feature labels on other segments!
        """
    if self.label is not None:
        label_x = 0.5 * (self.start_x_position + self.end_x_position) + (self.chr_percent + 0.05) * (self.end_x_position - self.start_x_position)
        label_y = (self.start_y_position - self.end_y_position) / 2 + self.end_y_position
        label_string = String(label_x, label_y, self.label)
        label_string.fontName = 'Helvetica'
        label_string.fontSize = self.label_size
        cur_drawing.add(label_string)