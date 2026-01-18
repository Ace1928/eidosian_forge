from the canonical composite structure in that we don't really have
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.graphics.shapes import Drawing, String, Line, Rect, Wedge, ArcPath
from reportlab.graphics.widgetbase import Widget
from Bio.Graphics import _write
from Bio.Graphics.GenomeDiagram import _Colors
class Chromosome(_ChromosomeComponent):
    """Class for drawing a chromosome of an organism.

    This organizes the drawing of a single organisms chromosome. This
    class can be instantiated directly, but the draw method makes the
    most sense to be called in the context of an organism.
    """

    def __init__(self, chromosome_name):
        """Initialize a Chromosome for drawing.

        Arguments:
         - chromosome_name - The label for the chromosome.

        Attributes:
         - start_x_position, end_x_position - The x positions on the page
           where the chromosome should be drawn. This allows multiple
           chromosomes to be drawn on a single page.
         - start_y_position, end_y_position - The y positions on the page
           where the chromosome should be contained.

        Configuration Attributes:
         - title_size - The size of the chromosome title.
         - scale_num - A number of scale the drawing by. This is useful if
           you want to draw multiple chromosomes of different sizes at the
           same scale. If this is not set, then the chromosome drawing will
           be scaled by the number of segments in the chromosome (so each
           chromosome will be the exact same final size).

        """
        _ChromosomeComponent.__init__(self)
        self._name = chromosome_name
        self.start_x_position = -1
        self.end_x_position = -1
        self.start_y_position = -1
        self.end_y_position = -1
        self.title_size = 20
        self.scale_num = None
        self.label_size = 6
        self.chr_percent = 0.25
        self.label_sep_percent = self.chr_percent * 0.5
        self._color_labels = False

    def subcomponent_size(self):
        """Return the scaled size of all subcomponents of this component."""
        total_sub = 0
        for sub_component in self._sub_components:
            total_sub += sub_component.scale
        return total_sub

    def draw(self, cur_drawing):
        """Draw a chromosome on the specified template.

        Ideally, the x_position and y_*_position attributes should be
        set prior to drawing -- otherwise we're going to have some problems.
        """
        for position in (self.start_x_position, self.end_x_position, self.start_y_position, self.end_y_position):
            assert position != -1, 'Need to set drawing coordinates.'
        cur_y_pos = self.start_y_position
        if self.scale_num:
            y_pos_change = (self.start_y_position * 0.95 - self.end_y_position) / self.scale_num
        elif len(self._sub_components) > 0:
            y_pos_change = (self.start_y_position * 0.95 - self.end_y_position) / self.subcomponent_size()
        else:
            pass
        left_labels = []
        right_labels = []
        for sub_component in self._sub_components:
            this_y_pos_change = sub_component.scale * y_pos_change
            sub_component.start_x_position = self.start_x_position
            sub_component.end_x_position = self.end_x_position
            sub_component.start_y_position = cur_y_pos
            sub_component.end_y_position = cur_y_pos - this_y_pos_change
            sub_component._left_labels = []
            sub_component._right_labels = []
            sub_component.draw(cur_drawing)
            left_labels += sub_component._left_labels
            right_labels += sub_component._right_labels
            cur_y_pos -= this_y_pos_change
        self._draw_labels(cur_drawing, left_labels, right_labels)
        self._draw_label(cur_drawing, self._name)

    def _draw_label(self, cur_drawing, label_name):
        """Draw a label for the chromosome (PRIVATE)."""
        x_position = 0.5 * (self.start_x_position + self.end_x_position)
        y_position = self.end_y_position
        label_string = String(x_position, y_position, label_name)
        label_string.fontName = 'Times-BoldItalic'
        label_string.fontSize = self.title_size
        label_string.textAnchor = 'middle'
        cur_drawing.add(label_string)

    def _draw_labels(self, cur_drawing, left_labels, right_labels):
        """Layout and draw sub-feature labels for the chromosome (PRIVATE).

        Tries to place each label at the same vertical position as the
        feature it applies to, but will adjust the positions to avoid or
        at least reduce label overlap.

        Draws the label text and a coloured line linking it to the
        location (i.e. feature) it applies to.
        """
        if not self._sub_components:
            return
        color_label = self._color_labels
        segment_width = (self.end_x_position - self.start_x_position) * self.chr_percent
        label_sep = (self.end_x_position - self.start_x_position) * self.label_sep_percent
        segment_x = self.start_x_position + 0.5 * (self.end_x_position - self.start_x_position - segment_width)
        y_limits = []
        for sub_component in self._sub_components:
            y_limits.extend((sub_component.start_y_position, sub_component.end_y_position))
        y_min = min(y_limits)
        y_max = max(y_limits)
        del y_limits
        h = self.label_size
        for x1, x2, labels, anchor in [(segment_x, segment_x - label_sep, _place_labels(left_labels, y_min, y_max, h), 'end'), (segment_x + segment_width, segment_x + segment_width + label_sep, _place_labels(right_labels, y_min, y_max, h), 'start')]:
            for y1, y2, color, back_color, name in labels:
                cur_drawing.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=0.25))
                label_string = String(x2, y2, name, textAnchor=anchor)
                label_string.fontName = 'Helvetica'
                label_string.fontSize = h
                if color_label:
                    label_string.fillColor = color
                if back_color:
                    w = stringWidth(name, label_string.fontName, label_string.fontSize)
                    if x1 > x2:
                        w = w * -1.0
                    cur_drawing.add(Rect(x2, y2 - 0.1 * h, w, h, strokeColor=back_color, fillColor=back_color))
                cur_drawing.add(label_string)