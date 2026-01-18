import warnings
import matplotlib as mpl
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd
from . import cm
from .axisgrid import Grid
from ._compat import get_colormap
from .utils import (
def _convert_colors(colors):
    """Convert either a list of colors or nested lists of colors to RGB."""
    to_rgb = mpl.colors.to_rgb
    try:
        to_rgb(colors[0])
        return list(map(to_rgb, colors))
    except ValueError:
        return [list(map(to_rgb, color_list)) for color_list in colors]