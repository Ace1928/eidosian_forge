import os
import inspect
import warnings
import colorsys
from contextlib import contextmanager
from urllib.request import urlopen, urlretrieve
from types import ModuleType
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.colors import to_rgb
import matplotlib.pyplot as plt
from matplotlib.cbook import normalize_kwargs
from seaborn._core.typing import deprecated
from seaborn.external.version import Version
from seaborn.external.appdirs import user_cache_dir
def _get_patch_legend_artist(fill):

    def legend_artist(**kws):
        color = kws.pop('color', None)
        if color is not None:
            if fill:
                kws['facecolor'] = color
            else:
                kws['edgecolor'] = color
                kws['facecolor'] = 'none'
        return mpl.patches.Rectangle((0, 0), 0, 0, **kws)
    return legend_artist