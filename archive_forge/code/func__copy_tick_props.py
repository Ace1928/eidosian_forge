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
def _copy_tick_props(self, src, dest):
    """Copy the props from src tick to dest tick."""
    if src is None or dest is None:
        return
    super()._copy_tick_props(src, dest)
    trans = dest._get_text1_transform()[0]
    dest.label1.set_transform(trans + dest._text1_translate)
    trans = dest._get_text2_transform()[0]
    dest.label2.set_transform(trans + dest._text2_translate)