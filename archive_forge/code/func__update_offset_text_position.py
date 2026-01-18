import datetime
import functools
import logging
from numbers import Real
import warnings
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
import matplotlib.artist as martist
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.scale as mscale
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import matplotlib.units as munits
def _update_offset_text_position(self, bboxes, bboxes2):
    """
        Update the offset_text position based on the sequence of bounding
        boxes of all the ticklabels
        """
    x, _ = self.offsetText.get_position()
    if 'outline' in self.axes.spines:
        bbox = self.axes.spines['outline'].get_window_extent()
    else:
        bbox = self.axes.bbox
    top = bbox.ymax
    self.offsetText.set_position((x, top + self.OFFSETTEXTPAD * self.figure.dpi / 72))