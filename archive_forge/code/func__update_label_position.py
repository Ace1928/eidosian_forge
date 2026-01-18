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
def _update_label_position(self, renderer):
    """
        Update the label position based on the bounding box enclosing
        all the ticklabels and axis spine
        """
    if not self._autolabelpos:
        return
    bboxes, bboxes2 = self._get_tick_boxes_siblings(renderer=renderer)
    x, y = self.label.get_position()
    if self.label_position == 'left':
        try:
            spine = self.axes.spines['left']
            spinebbox = spine.get_window_extent()
        except KeyError:
            spinebbox = self.axes.bbox
        bbox = mtransforms.Bbox.union(bboxes + [spinebbox])
        left = bbox.x0
        self.label.set_position((left - self.labelpad * self.figure.dpi / 72, y))
    else:
        try:
            spine = self.axes.spines['right']
            spinebbox = spine.get_window_extent()
        except KeyError:
            spinebbox = self.axes.bbox
        bbox = mtransforms.Bbox.union(bboxes2 + [spinebbox])
        right = bbox.x1
        self.label.set_position((right + self.labelpad * self.figure.dpi / 72, y))