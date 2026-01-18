from the canonical composite structure in that we don't really have
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.graphics.shapes import Drawing, String, Line, Rect, Wedge, ArcPath
from reportlab.graphics.widgetbase import Widget
from Bio.Graphics import _write
from Bio.Graphics.GenomeDiagram import _Colors
def _spring_layout(desired, minimum, maximum, gap=0):
    """Try to layout label coordinates or other floats (PRIVATE).

    Originally written for the y-axis vertical positioning of labels on a
    chromosome diagram (where the minimum gap between y-axis coordinates is
    the label height), it could also potentially be used for x-axis placement,
    or indeed radial placement for circular chromosomes within GenomeDiagram.

    In essence this is an optimisation problem, balancing the desire to have
    each label as close as possible to its data point, but also to spread out
    the labels to avoid overlaps. This could be described with a cost function
    (modelling the label distance from the desired placement, and the inter-
    label separations as springs) and solved as a multi-variable minimization
    problem - perhaps with NumPy or SciPy.

    For now however, the implementation is a somewhat crude ad hoc algorithm.

    NOTE - This expects the input data to have been sorted!
    """
    count = len(desired)
    if count <= 1:
        return desired
    if minimum >= maximum:
        raise ValueError(f'Bad min/max {minimum:f} and {maximum:f}')
    if min(desired) < minimum or max(desired) > maximum:
        raise ValueError('Data %f to %f out of bounds (%f to %f)' % (min(desired), max(desired), minimum, maximum))
    equal_step = (maximum - minimum) / (count - 1)
    if equal_step < gap:
        import warnings
        from Bio import BiopythonWarning
        warnings.warn('Too many labels to avoid overlap', BiopythonWarning)
        return [minimum + i * equal_step for i in range(count)]
    good = True
    if gap:
        prev = desired[0]
        for next in desired[1:]:
            if prev - next < gap:
                good = False
                break
    if good:
        return desired
    span = maximum - minimum
    for split in [0.5 * span, span / 3.0, 2 * span / 3.0, 0.25 * span, 0.75 * span]:
        midpoint = minimum + split
        low = [x for x in desired if x <= midpoint - 0.5 * gap]
        high = [x for x in desired if x > midpoint + 0.5 * gap]
        if len(low) + len(high) < count:
            continue
        elif not low and len(high) * gap <= span - split + 0.5 * gap:
            return _spring_layout(high, midpoint + 0.5 * gap, maximum, gap)
        elif not high and len(low) * gap <= split + 0.5 * gap:
            return _spring_layout(low, minimum, midpoint - 0.5 * gap, gap)
        elif len(low) * gap <= split - 0.5 * gap and len(high) * gap <= span - split - 0.5 * gap:
            return _spring_layout(low, minimum, midpoint - 0.5 * gap, gap) + _spring_layout(high, midpoint + 0.5 * gap, maximum, gap)
    low = min(desired)
    high = max(desired)
    if (high - low) / (count - 1) >= gap:
        equal_step = (high - low) / (count - 1)
        return [low + i * equal_step for i in range(count)]
    low = 0.5 * (minimum + min(desired))
    high = 0.5 * (max(desired) + maximum)
    if (high - low) / (count - 1) >= gap:
        equal_step = (high - low) / (count - 1)
        return [low + i * equal_step for i in range(count)]
    return [minimum + i * equal_step for i in range(count)]