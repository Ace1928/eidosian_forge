import math
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing, String
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.barcharts import BarChartProperties
from reportlab.graphics.widgetbase import TypedPropertyCollection
from Bio.Graphics import _write
class DistributionPage:
    """Display a grouping of distributions on a page.

    This organizes Distributions, and will display them nicely
    on a single page.
    """

    def __init__(self, output_format='pdf'):
        """Initialize the class."""
        self.distributions = []
        self.number_of_columns = 1
        self.page_size = letter
        self.title_size = 20
        self.output_format = output_format

    def draw(self, output_file, title):
        """Draw out the distribution information.

        Arguments:
         - output_file - The name of the file to output the information to,
           or a handle to write to.
         - title - A title to display on the graphic.

        """
        width, height = self.page_size
        cur_drawing = Drawing(width, height)
        self._draw_title(cur_drawing, title, width, height)
        cur_x_pos = inch * 0.5
        end_x_pos = width - inch * 0.5
        cur_y_pos = height - 1.5 * inch
        end_y_pos = 0.5 * inch
        x_pos_change = (end_x_pos - cur_x_pos) / self.number_of_columns
        num_y_rows = math.ceil(len(self.distributions) / self.number_of_columns)
        y_pos_change = (cur_y_pos - end_y_pos) / num_y_rows
        self._draw_distributions(cur_drawing, cur_x_pos, x_pos_change, cur_y_pos, y_pos_change, num_y_rows)
        self._draw_legend(cur_drawing, 2.5 * inch, width)
        return _write(cur_drawing, output_file, self.output_format)

    def _draw_title(self, cur_drawing, title, width, height):
        """Add the title of the figure to the drawing (PRIVATE)."""
        title_string = String(width / 2, height - inch, title)
        title_string.fontName = 'Helvetica-Bold'
        title_string.fontSize = self.title_size
        title_string.textAnchor = 'middle'
        cur_drawing.add(title_string)

    def _draw_distributions(self, cur_drawing, start_x_pos, x_pos_change, start_y_pos, y_pos_change, num_y_drawings):
        """Draw all of the distributions on the page (PRIVATE).

        Arguments:
         - cur_drawing - The drawing we are working with.
         - start_x_pos - The x position on the page to start drawing at.
         - x_pos_change - The change in x position between each figure.
         - start_y_pos - The y position on the page to start drawing at.
         - y_pos_change - The change in y position between each figure.
         - num_y_drawings - The number of drawings we'll have in the y
           (up/down) direction.

        """
        for y_drawing in range(int(num_y_drawings)):
            if (y_drawing + 1) * self.number_of_columns > len(self.distributions):
                num_x_drawings = len(self.distributions) - y_drawing * self.number_of_columns
            else:
                num_x_drawings = self.number_of_columns
            for x_drawing in range(num_x_drawings):
                dist_num = y_drawing * self.number_of_columns + x_drawing
                cur_distribution = self.distributions[dist_num]
                x_pos = start_x_pos + x_drawing * x_pos_change
                end_x_pos = x_pos + x_pos_change
                end_y_pos = start_y_pos - y_drawing * y_pos_change
                y_pos = end_y_pos - y_pos_change
                cur_distribution.draw(cur_drawing, x_pos, y_pos, end_x_pos, end_y_pos)

    def _draw_legend(self, cur_drawing, start_y, width):
        """Add a legend to the figure (PRIVATE).

        Subclasses can implement to provide a specialized legend.
        """