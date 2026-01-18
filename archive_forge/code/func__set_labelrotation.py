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
def _set_labelrotation(self, labelrotation):
    if isinstance(labelrotation, str):
        mode = labelrotation
        angle = 0
    elif isinstance(labelrotation, (tuple, list)):
        mode, angle = labelrotation
    else:
        mode = 'default'
        angle = labelrotation
    _api.check_in_list(['auto', 'default'], labelrotation=mode)
    self._labelrotation = (mode, angle)