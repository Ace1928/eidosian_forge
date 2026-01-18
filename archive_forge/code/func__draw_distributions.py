import math
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing, String
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.barcharts import BarChartProperties
from reportlab.graphics.widgetbase import TypedPropertyCollection
from Bio.Graphics import _write
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