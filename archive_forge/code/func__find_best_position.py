import itertools
import logging
import numbers
import time
import numpy as np
import matplotlib as mpl
from matplotlib import _api, _docstring, colors, offsetbox
from matplotlib.artist import Artist, allow_rasterization
from matplotlib.cbook import silent_list
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import (Patch, Rectangle, Shadow, FancyBboxPatch,
from matplotlib.collections import (
from matplotlib.text import Text
from matplotlib.transforms import Bbox, BboxBase, TransformedBbox
from matplotlib.transforms import BboxTransformTo, BboxTransformFrom
from matplotlib.offsetbox import (
from matplotlib.container import ErrorbarContainer, BarContainer, StemContainer
from . import legend_handler
def _find_best_position(self, width, height, renderer, consider=None):
    """
        Determine the best location to place the legend.

        *consider* is a list of ``(x, y)`` pairs to consider as a potential
        lower-left corner of the legend. All are display coords.
        """
    assert self.isaxes
    start_time = time.perf_counter()
    bboxes, lines, offsets = self._auto_legend_data()
    bbox = Bbox.from_bounds(0, 0, width, height)
    if consider is None:
        consider = [self._get_anchored_bbox(x, bbox, self.get_bbox_to_anchor(), renderer) for x in range(1, len(self.codes))]
    candidates = []
    for idx, (l, b) in enumerate(consider):
        legendBox = Bbox.from_bounds(l, b, width, height)
        badness = 0
        badness = sum((legendBox.count_contains(line.vertices) for line in lines)) + legendBox.count_contains(offsets) + legendBox.count_overlaps(bboxes) + sum((line.intersects_bbox(legendBox, filled=False) for line in lines))
        if badness == 0:
            return (l, b)
        candidates.append((badness, idx, (l, b)))
    _, _, (l, b) = min(candidates)
    if self._loc_used_default and time.perf_counter() - start_time > 1:
        _api.warn_external('Creating legend with loc="best" can be slow with large amounts of data.')
    return (l, b)