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
def _calculate_linkage_fastcluster(self):
    import fastcluster
    euclidean_methods = ('centroid', 'median', 'ward')
    euclidean = self.metric == 'euclidean' and self.method in euclidean_methods
    if euclidean or self.method == 'single':
        return fastcluster.linkage_vector(self.array, method=self.method, metric=self.metric)
    else:
        linkage = fastcluster.linkage(self.array, method=self.method, metric=self.metric)
        return linkage