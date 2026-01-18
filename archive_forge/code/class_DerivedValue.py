import codecs, re
from reportlab.lib.utils import isSeq, isBytes, isStr
from reportlab.lib import colors
from reportlab.lib.normalDate import NormalDate
class DerivedValue:
    """This is used for magic values which work themselves out.
    An example would be an "inherit" property, so that one can have

      drawing.chart.categoryAxis.labels.fontName = inherit

    and pick up the value from the top of the drawing.
    Validators will permit this provided that a value can be pulled
    in which satisfies it.  And the renderer will have special
    knowledge of these so they can evaluate themselves.
    """

    def getValue(self, renderer, attr):
        """Override this.  The renderers will pass the renderer,
        and the attribute name.  Algorithms can then backtrack up
        through all the stuff the renderer provides, including
        a correct stack of parent nodes."""
        return None