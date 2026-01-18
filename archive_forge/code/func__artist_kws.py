from numbers import Number
from functools import partial
import math
import textwrap
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as tx
from matplotlib.cbook import normalize_kwargs
from matplotlib.colors import to_rgba
from matplotlib.collections import LineCollection
from ._base import VectorPlotter
from ._statistics import ECDF, Histogram, KDE
from ._stats.counting import Hist
from .axisgrid import (
from .utils import (
from .palettes import color_palette
from .external import husl
from .external.kde import gaussian_kde
from ._docstrings import (
important parameter. Misspecification of the bandwidth can produce a
def _artist_kws(self, kws, fill, element, multiple, color, alpha):
    """Handle differences between artists in filled/unfilled plots."""
    kws = kws.copy()
    if fill:
        kws = normalize_kwargs(kws, mpl.collections.PolyCollection)
        kws.setdefault('facecolor', to_rgba(color, alpha))
        if element == 'bars':
            kws['color'] = 'none'
        if multiple in ['stack', 'fill'] or element == 'bars':
            kws.setdefault('edgecolor', mpl.rcParams['patch.edgecolor'])
        else:
            kws.setdefault('edgecolor', to_rgba(color, 1))
    elif element == 'bars':
        kws['facecolor'] = 'none'
        kws['edgecolor'] = to_rgba(color, alpha)
    else:
        kws['color'] = to_rgba(color, alpha)
    return kws