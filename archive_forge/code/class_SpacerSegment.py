from the canonical composite structure in that we don't really have
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.graphics.shapes import Drawing, String, Line, Rect, Wedge, ArcPath
from reportlab.graphics.widgetbase import Widget
from Bio.Graphics import _write
from Bio.Graphics.GenomeDiagram import _Colors
class SpacerSegment(ChromosomeSegment):
    """A segment that is located at the end of a linear chromosome.

    Doesn't draw anything, just empty space which can be helpful
    for layout purposes (e.g. making room for feature labels).
    """

    def draw(self, cur_diagram):
        """Draw nothing to the current diagram (dummy method).

        The segment spacer has no actual image in the diagram,
        so this method therefore does nothing, but is defined
        to match the expected API of the other segment objects.
        """