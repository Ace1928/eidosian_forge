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
def _get_legend_handles_labels(axs, legend_handler_map=None):
    """Return handles and labels for legend."""
    handles = []
    labels = []
    for handle in _get_legend_handles(axs, legend_handler_map):
        label = handle.get_label()
        if label and (not label.startswith('_')):
            handles.append(handle)
            labels.append(label)
    return (handles, labels)