import math
import types
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
from matplotlib.axes import Axes
import matplotlib.axis as maxis
import matplotlib.markers as mmarkers
import matplotlib.patches as mpatches
from matplotlib.path import Path
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
from matplotlib.spines import Spine
def _update_padding(self, pad, angle):
    padx = pad * np.cos(angle) / 72
    pady = pad * np.sin(angle) / 72
    self._text1_translate._t = (padx, pady)
    self._text1_translate.invalidate()
    self._text2_translate._t = (-padx, -pady)
    self._text2_translate.invalidate()