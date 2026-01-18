from __future__ import division
import datetime
import math
class WidgetHFill(Widget):
    """The base class for all variable width widgets.

    This widget is much like the \\hfill command in TeX, it will expand to
    fill the line. You can use more than one in the same line, and they will
    all have the same width, and together will fill the line.
    """

    @abstractmethod
    def update(self, pbar, width):
        """Updates the widget providing the total width the widget must fill.

        pbar - a reference to the calling ProgressBar
        width - The total width the widget must fill
        """