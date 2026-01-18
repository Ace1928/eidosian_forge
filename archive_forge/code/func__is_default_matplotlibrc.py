from .. import utils
from .._lazyload import matplotlib as mpl
from .._lazyload import mpl_toolkits
import numpy as np
import platform
def _is_default_matplotlibrc():
    __defaults = {'axes.labelsize': 'medium', 'axes.titlesize': 'large', 'figure.titlesize': 'large', 'legend.fontsize': 'medium', 'legend.title_fontsize': None, 'xtick.labelsize': 'medium', 'ytick.labelsize': 'medium'}
    for k, v in __defaults.items():
        if plt.rcParams[k] != v:
            return False
    return True