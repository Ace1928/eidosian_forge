from the canonical composite structure in that we don't really have
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.graphics.shapes import Drawing, String, Line, Rect, Wedge, ArcPath
from reportlab.graphics.widgetbase import Widget
from Bio.Graphics import _write
from Bio.Graphics.GenomeDiagram import _Colors
def _place_labels(desired_etc, minimum, maximum, gap=0):
    desired_etc.sort()
    placed = _spring_layout([row[0] for row in desired_etc], minimum, maximum, gap)
    for old, y2 in zip(desired_etc, placed):
        yield ((old[0], y2) + tuple(old[1:]))