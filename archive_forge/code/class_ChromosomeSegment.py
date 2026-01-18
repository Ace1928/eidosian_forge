from the canonical composite structure in that we don't really have
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.graphics.shapes import Drawing, String, Line, Rect, Wedge, ArcPath
from reportlab.graphics.widgetbase import Widget
from Bio.Graphics import _write
from Bio.Graphics.GenomeDiagram import _Colors
class ChromosomeSegment(_ChromosomeComponent):
    """Draw a segment of a chromosome.

    This class provides the important configurable functionality of drawing
    a Chromosome. Each segment has some customization available here, or can
    be subclassed to define additional functionality. Most of the interesting
    drawing stuff is likely to happen at the ChromosomeSegment level.
    """

    def __init__(self):
        """Initialize a ChromosomeSegment.

        Attributes:
         - start_x_position, end_x_position - Defines the x range we have
           to draw things in.
         - start_y_position, end_y_position - Defines the y range we have
           to draw things in.

        Configuration Attributes:
         - scale - A scaling value for the component. By default this is
           set at 1 (ie -- has the same scale as everything else). Higher
           values give more size to the component, smaller values give less.
         - fill_color - A color to fill in the segment with. Colors are
           available in reportlab.lib.colors
         - label - A label to place on the chromosome segment. This should
           be a text string specifying what is to be included in the label.
         - label_size - The size of the label.
         - chr_percent - The percentage of area that the chromosome
           segment takes up.

        """
        _ChromosomeComponent.__init__(self)
        self.start_x_position = -1
        self.end_x_position = -1
        self.start_y_position = -1
        self.end_y_position = -1
        self.scale = 1
        self.fill_color = None
        self.label = None
        self.label_size = 6
        self.chr_percent = 0.25

    def draw(self, cur_drawing):
        """Draw a chromosome segment.

        Before drawing, the range we are drawing in needs to be set.
        """
        for position in (self.start_x_position, self.end_x_position, self.start_y_position, self.end_y_position):
            assert position != -1, 'Need to set drawing coordinates.'
        self._draw_subcomponents(cur_drawing)
        self._draw_segment(cur_drawing)
        self._overdraw_subcomponents(cur_drawing)
        self._draw_label(cur_drawing)

    def _draw_subcomponents(self, cur_drawing):
        """Draw any subcomponents of the chromosome segment (PRIVATE).

        This should be overridden in derived classes if there are
        subcomponents to be drawn.
        """

    def _draw_segment(self, cur_drawing):
        """Draw the current chromosome segment (PRIVATE)."""
        segment_y = self.end_y_position
        segment_width = (self.end_x_position - self.start_x_position) * self.chr_percent
        segment_height = self.start_y_position - self.end_y_position
        segment_x = self.start_x_position + 0.5 * (self.end_x_position - self.start_x_position - segment_width)
        right_line = Line(segment_x, segment_y, segment_x, segment_y + segment_height)
        left_line = Line(segment_x + segment_width, segment_y, segment_x + segment_width, segment_y + segment_height)
        cur_drawing.add(right_line)
        cur_drawing.add(left_line)
        if self.fill_color is not None:
            fill_rectangle = Rect(segment_x, segment_y, segment_width, segment_height)
            fill_rectangle.fillColor = self.fill_color
            fill_rectangle.strokeColor = None
            cur_drawing.add(fill_rectangle)

    def _overdraw_subcomponents(self, cur_drawing):
        """Draw any subcomponents of the chromosome segment over the main part (PRIVATE).

        This should be overridden in derived classes if there are
        subcomponents to be drawn.
        """

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