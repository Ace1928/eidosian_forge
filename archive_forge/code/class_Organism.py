from the canonical composite structure in that we don't really have
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.graphics.shapes import Drawing, String, Line, Rect, Wedge, ArcPath
from reportlab.graphics.widgetbase import Widget
from Bio.Graphics import _write
from Bio.Graphics.GenomeDiagram import _Colors
class Organism(_ChromosomeComponent):
    """Top level class for drawing chromosomes.

    This class holds information about an organism and all of its
    chromosomes, and provides the top level object which could be used
    for drawing a chromosome representation of an organism.

    Chromosomes should be added and removed from the Organism via the
    add and remove functions.
    """

    def __init__(self, output_format='pdf'):
        """Initialize the class."""
        _ChromosomeComponent.__init__(self)
        self.page_size = letter
        self.title_size = 20
        self._legend_height = 0
        self.output_format = output_format

    def draw(self, output_file, title):
        """Draw out the information for the Organism.

        Arguments:
         - output_file -- The name of a file specifying where the
           document should be saved, or a handle to be written to.
           The output format is set when creating the Organism object.
           Alternatively, output_file=None will return the drawing using
           the low-level ReportLab objects (for further processing, such
           as adding additional graphics, before writing).
         - title -- The output title of the produced document.

        """
        width, height = self.page_size
        cur_drawing = Drawing(width, height)
        self._draw_title(cur_drawing, title, width, height)
        cur_x_pos = inch * 0.5
        if len(self._sub_components) > 0:
            x_pos_change = (width - inch) / len(self._sub_components)
        else:
            pass
        for sub_component in self._sub_components:
            sub_component.start_x_position = cur_x_pos + 0.05 * x_pos_change
            sub_component.end_x_position = cur_x_pos + 0.95 * x_pos_change
            sub_component.start_y_position = height - 1.5 * inch
            sub_component.end_y_position = self._legend_height + 1 * inch
            sub_component.draw(cur_drawing)
            cur_x_pos += x_pos_change
        self._draw_legend(cur_drawing, self._legend_height + 0.5 * inch, width)
        if output_file is None:
            return cur_drawing
        return _write(cur_drawing, output_file, self.output_format)

    def _draw_title(self, cur_drawing, title, width, height):
        """Write out the title of the organism figure (PRIVATE)."""
        title_string = String(width / 2, height - inch, title)
        title_string.fontName = 'Helvetica-Bold'
        title_string.fontSize = self.title_size
        title_string.textAnchor = 'middle'
        cur_drawing.add(title_string)

    def _draw_legend(self, cur_drawing, start_y, width):
        """Draw a legend for the figure (PRIVATE).

        Subclasses should implement this (see also self._legend_height) to
        provide specialized legends.
        """