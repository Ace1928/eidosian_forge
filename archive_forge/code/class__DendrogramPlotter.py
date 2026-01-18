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
class _DendrogramPlotter:
    """Object for drawing tree of similarities between data rows/columns"""

    def __init__(self, data, linkage, metric, method, axis, label, rotate):
        """Plot a dendrogram of the relationships between the columns of data

        Parameters
        ----------
        data : pandas.DataFrame
            Rectangular data
        """
        self.axis = axis
        if self.axis == 1:
            data = data.T
        if isinstance(data, pd.DataFrame):
            array = data.values
        else:
            array = np.asarray(data)
            data = pd.DataFrame(array)
        self.array = array
        self.data = data
        self.shape = self.data.shape
        self.metric = metric
        self.method = method
        self.axis = axis
        self.label = label
        self.rotate = rotate
        if linkage is None:
            self.linkage = self.calculated_linkage
        else:
            self.linkage = linkage
        self.dendrogram = self.calculate_dendrogram()
        ticks = 10 * np.arange(self.data.shape[0]) + 5
        if self.label:
            ticklabels = _index_to_ticklabels(self.data.index)
            ticklabels = [ticklabels[i] for i in self.reordered_ind]
            if self.rotate:
                self.xticks = []
                self.yticks = ticks
                self.xticklabels = []
                self.yticklabels = ticklabels
                self.ylabel = _index_to_label(self.data.index)
                self.xlabel = ''
            else:
                self.xticks = ticks
                self.yticks = []
                self.xticklabels = ticklabels
                self.yticklabels = []
                self.ylabel = ''
                self.xlabel = _index_to_label(self.data.index)
        else:
            self.xticks, self.yticks = ([], [])
            self.yticklabels, self.xticklabels = ([], [])
            self.xlabel, self.ylabel = ('', '')
        self.dependent_coord = self.dendrogram['dcoord']
        self.independent_coord = self.dendrogram['icoord']

    def _calculate_linkage_scipy(self):
        linkage = hierarchy.linkage(self.array, method=self.method, metric=self.metric)
        return linkage

    def _calculate_linkage_fastcluster(self):
        import fastcluster
        euclidean_methods = ('centroid', 'median', 'ward')
        euclidean = self.metric == 'euclidean' and self.method in euclidean_methods
        if euclidean or self.method == 'single':
            return fastcluster.linkage_vector(self.array, method=self.method, metric=self.metric)
        else:
            linkage = fastcluster.linkage(self.array, method=self.method, metric=self.metric)
            return linkage

    @property
    def calculated_linkage(self):
        try:
            return self._calculate_linkage_fastcluster()
        except ImportError:
            if np.prod(self.shape) >= 10000:
                msg = 'Clustering large matrix with scipy. Installing `fastcluster` may give better performance.'
                warnings.warn(msg)
        return self._calculate_linkage_scipy()

    def calculate_dendrogram(self):
        """Calculates a dendrogram based on the linkage matrix

        Made a separate function, not a property because don't want to
        recalculate the dendrogram every time it is accessed.

        Returns
        -------
        dendrogram : dict
            Dendrogram dictionary as returned by scipy.cluster.hierarchy
            .dendrogram. The important key-value pairing is
            "reordered_ind" which indicates the re-ordering of the matrix
        """
        return hierarchy.dendrogram(self.linkage, no_plot=True, color_threshold=-np.inf)

    @property
    def reordered_ind(self):
        """Indices of the matrix, reordered by the dendrogram"""
        return self.dendrogram['leaves']

    def plot(self, ax, tree_kws):
        """Plots a dendrogram of the similarities between data on the axes

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes object upon which the dendrogram is plotted

        """
        tree_kws = {} if tree_kws is None else tree_kws.copy()
        tree_kws.setdefault('linewidths', 0.5)
        tree_kws.setdefault('colors', tree_kws.pop('color', (0.2, 0.2, 0.2)))
        if self.rotate and self.axis == 0:
            coords = zip(self.dependent_coord, self.independent_coord)
        else:
            coords = zip(self.independent_coord, self.dependent_coord)
        lines = LineCollection([list(zip(x, y)) for x, y in coords], **tree_kws)
        ax.add_collection(lines)
        number_of_leaves = len(self.reordered_ind)
        max_dependent_coord = max(map(max, self.dependent_coord))
        if self.rotate:
            ax.yaxis.set_ticks_position('right')
            ax.set_ylim(0, number_of_leaves * 10)
            ax.set_xlim(0, max_dependent_coord * 1.05)
            ax.invert_xaxis()
            ax.invert_yaxis()
        else:
            ax.set_xlim(0, number_of_leaves * 10)
            ax.set_ylim(0, max_dependent_coord * 1.05)
        despine(ax=ax, bottom=True, left=True)
        ax.set(xticks=self.xticks, yticks=self.yticks, xlabel=self.xlabel, ylabel=self.ylabel)
        xtl = ax.set_xticklabels(self.xticklabels)
        ytl = ax.set_yticklabels(self.yticklabels, rotation='vertical')
        _draw_figure(ax.figure)
        if len(ytl) > 0 and axis_ticklabels_overlap(ytl):
            plt.setp(ytl, rotation='horizontal')
        if len(xtl) > 0 and axis_ticklabels_overlap(xtl):
            plt.setp(xtl, rotation='vertical')
        return self