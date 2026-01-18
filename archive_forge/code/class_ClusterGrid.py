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
class ClusterGrid(Grid):

    def __init__(self, data, pivot_kws=None, z_score=None, standard_scale=None, figsize=None, row_colors=None, col_colors=None, mask=None, dendrogram_ratio=None, colors_ratio=None, cbar_pos=None):
        """Grid object for organizing clustered heatmap input on to axes"""
        if _no_scipy:
            raise RuntimeError('ClusterGrid requires scipy to be available')
        if isinstance(data, pd.DataFrame):
            self.data = data
        else:
            self.data = pd.DataFrame(data)
        self.data2d = self.format_data(self.data, pivot_kws, z_score, standard_scale)
        self.mask = _matrix_mask(self.data2d, mask)
        self._figure = plt.figure(figsize=figsize)
        self.row_colors, self.row_color_labels = self._preprocess_colors(data, row_colors, axis=0)
        self.col_colors, self.col_color_labels = self._preprocess_colors(data, col_colors, axis=1)
        try:
            row_dendrogram_ratio, col_dendrogram_ratio = dendrogram_ratio
        except TypeError:
            row_dendrogram_ratio = col_dendrogram_ratio = dendrogram_ratio
        try:
            row_colors_ratio, col_colors_ratio = colors_ratio
        except TypeError:
            row_colors_ratio = col_colors_ratio = colors_ratio
        width_ratios = self.dim_ratios(self.row_colors, row_dendrogram_ratio, row_colors_ratio)
        height_ratios = self.dim_ratios(self.col_colors, col_dendrogram_ratio, col_colors_ratio)
        nrows = 2 if self.col_colors is None else 3
        ncols = 2 if self.row_colors is None else 3
        self.gs = gridspec.GridSpec(nrows, ncols, width_ratios=width_ratios, height_ratios=height_ratios)
        self.ax_row_dendrogram = self._figure.add_subplot(self.gs[-1, 0])
        self.ax_col_dendrogram = self._figure.add_subplot(self.gs[0, -1])
        self.ax_row_dendrogram.set_axis_off()
        self.ax_col_dendrogram.set_axis_off()
        self.ax_row_colors = None
        self.ax_col_colors = None
        if self.row_colors is not None:
            self.ax_row_colors = self._figure.add_subplot(self.gs[-1, 1])
        if self.col_colors is not None:
            self.ax_col_colors = self._figure.add_subplot(self.gs[1, -1])
        self.ax_heatmap = self._figure.add_subplot(self.gs[-1, -1])
        if cbar_pos is None:
            self.ax_cbar = self.cax = None
        else:
            self.ax_cbar = self._figure.add_subplot(self.gs[0, 0])
            self.cax = self.ax_cbar
        self.cbar_pos = cbar_pos
        self.dendrogram_row = None
        self.dendrogram_col = None

    def _preprocess_colors(self, data, colors, axis):
        """Preprocess {row/col}_colors to extract labels and convert colors."""
        labels = None
        if colors is not None:
            if isinstance(colors, (pd.DataFrame, pd.Series)):
                if not hasattr(data, 'index') and axis == 0 or (not hasattr(data, 'columns') and axis == 1):
                    axis_name = 'col' if axis else 'row'
                    msg = f"{axis_name}_colors indices can't be matched with data indices. Provide {axis_name}_colors as a non-indexed datatype, e.g. by using `.to_numpy()``"
                    raise TypeError(msg)
                if axis == 0:
                    colors = colors.reindex(data.index)
                else:
                    colors = colors.reindex(data.columns)
                colors = colors.astype(object).fillna('white')
                if isinstance(colors, pd.DataFrame):
                    labels = list(colors.columns)
                    colors = colors.T.values
                else:
                    if colors.name is None:
                        labels = ['']
                    else:
                        labels = [colors.name]
                    colors = colors.values
            colors = _convert_colors(colors)
        return (colors, labels)

    def format_data(self, data, pivot_kws, z_score=None, standard_scale=None):
        """Extract variables from data or use directly."""
        if pivot_kws is not None:
            data2d = data.pivot(**pivot_kws)
        else:
            data2d = data
        if z_score is not None and standard_scale is not None:
            raise ValueError('Cannot perform both z-scoring and standard-scaling on data')
        if z_score is not None:
            data2d = self.z_score(data2d, z_score)
        if standard_scale is not None:
            data2d = self.standard_scale(data2d, standard_scale)
        return data2d

    @staticmethod
    def z_score(data2d, axis=1):
        """Standarize the mean and variance of the data axis

        Parameters
        ----------
        data2d : pandas.DataFrame
            Data to normalize
        axis : int
            Which axis to normalize across. If 0, normalize across rows, if 1,
            normalize across columns.

        Returns
        -------
        normalized : pandas.DataFrame
            Noramlized data with a mean of 0 and variance of 1 across the
            specified axis.
        """
        if axis == 1:
            z_scored = data2d
        else:
            z_scored = data2d.T
        z_scored = (z_scored - z_scored.mean()) / z_scored.std()
        if axis == 1:
            return z_scored
        else:
            return z_scored.T

    @staticmethod
    def standard_scale(data2d, axis=1):
        """Divide the data by the difference between the max and min

        Parameters
        ----------
        data2d : pandas.DataFrame
            Data to normalize
        axis : int
            Which axis to normalize across. If 0, normalize across rows, if 1,
            normalize across columns.

        Returns
        -------
        standardized : pandas.DataFrame
            Noramlized data with a mean of 0 and variance of 1 across the
            specified axis.

        """
        if axis == 1:
            standardized = data2d
        else:
            standardized = data2d.T
        subtract = standardized.min()
        standardized = (standardized - subtract) / (standardized.max() - standardized.min())
        if axis == 1:
            return standardized
        else:
            return standardized.T

    def dim_ratios(self, colors, dendrogram_ratio, colors_ratio):
        """Get the proportions of the figure taken up by each axes."""
        ratios = [dendrogram_ratio]
        if colors is not None:
            if np.ndim(colors) > 2:
                n_colors = len(colors)
            else:
                n_colors = 1
            ratios += [n_colors * colors_ratio]
        ratios.append(1 - sum(ratios))
        return ratios

    @staticmethod
    def color_list_to_matrix_and_cmap(colors, ind, axis=0):
        """Turns a list of colors into a numpy matrix and matplotlib colormap

        These arguments can now be plotted using heatmap(matrix, cmap)
        and the provided colors will be plotted.

        Parameters
        ----------
        colors : list of matplotlib colors
            Colors to label the rows or columns of a dataframe.
        ind : list of ints
            Ordering of the rows or columns, to reorder the original colors
            by the clustered dendrogram order
        axis : int
            Which axis this is labeling

        Returns
        -------
        matrix : numpy.array
            A numpy array of integer values, where each indexes into the cmap
        cmap : matplotlib.colors.ListedColormap

        """
        try:
            mpl.colors.to_rgb(colors[0])
        except ValueError:
            m, n = (len(colors), len(colors[0]))
            if not all((len(c) == n for c in colors[1:])):
                raise ValueError('Multiple side color vectors must have same size')
        else:
            m, n = (1, len(colors))
            colors = [colors]
        unique_colors = {}
        matrix = np.zeros((m, n), int)
        for i, inner in enumerate(colors):
            for j, color in enumerate(inner):
                idx = unique_colors.setdefault(color, len(unique_colors))
                matrix[i, j] = idx
        matrix = matrix[:, ind]
        if axis == 0:
            matrix = matrix.T
        cmap = mpl.colors.ListedColormap(list(unique_colors))
        return (matrix, cmap)

    def plot_dendrograms(self, row_cluster, col_cluster, metric, method, row_linkage, col_linkage, tree_kws):
        if row_cluster:
            self.dendrogram_row = dendrogram(self.data2d, metric=metric, method=method, label=False, axis=0, ax=self.ax_row_dendrogram, rotate=True, linkage=row_linkage, tree_kws=tree_kws)
        else:
            self.ax_row_dendrogram.set_xticks([])
            self.ax_row_dendrogram.set_yticks([])
        if col_cluster:
            self.dendrogram_col = dendrogram(self.data2d, metric=metric, method=method, label=False, axis=1, ax=self.ax_col_dendrogram, linkage=col_linkage, tree_kws=tree_kws)
        else:
            self.ax_col_dendrogram.set_xticks([])
            self.ax_col_dendrogram.set_yticks([])
        despine(ax=self.ax_row_dendrogram, bottom=True, left=True)
        despine(ax=self.ax_col_dendrogram, bottom=True, left=True)

    def plot_colors(self, xind, yind, **kws):
        """Plots color labels between the dendrogram and the heatmap

        Parameters
        ----------
        heatmap_kws : dict
            Keyword arguments heatmap

        """
        kws = kws.copy()
        kws.pop('cmap', None)
        kws.pop('norm', None)
        kws.pop('center', None)
        kws.pop('annot', None)
        kws.pop('vmin', None)
        kws.pop('vmax', None)
        kws.pop('robust', None)
        kws.pop('xticklabels', None)
        kws.pop('yticklabels', None)
        if self.row_colors is not None:
            matrix, cmap = self.color_list_to_matrix_and_cmap(self.row_colors, yind, axis=0)
            if self.row_color_labels is not None:
                row_color_labels = self.row_color_labels
            else:
                row_color_labels = False
            heatmap(matrix, cmap=cmap, cbar=False, ax=self.ax_row_colors, xticklabels=row_color_labels, yticklabels=False, **kws)
            if row_color_labels is not False:
                plt.setp(self.ax_row_colors.get_xticklabels(), rotation=90)
        else:
            despine(self.ax_row_colors, left=True, bottom=True)
        if self.col_colors is not None:
            matrix, cmap = self.color_list_to_matrix_and_cmap(self.col_colors, xind, axis=1)
            if self.col_color_labels is not None:
                col_color_labels = self.col_color_labels
            else:
                col_color_labels = False
            heatmap(matrix, cmap=cmap, cbar=False, ax=self.ax_col_colors, xticklabels=False, yticklabels=col_color_labels, **kws)
            if col_color_labels is not False:
                self.ax_col_colors.yaxis.tick_right()
                plt.setp(self.ax_col_colors.get_yticklabels(), rotation=0)
        else:
            despine(self.ax_col_colors, left=True, bottom=True)

    def plot_matrix(self, colorbar_kws, xind, yind, **kws):
        self.data2d = self.data2d.iloc[yind, xind]
        self.mask = self.mask.iloc[yind, xind]
        xtl = kws.pop('xticklabels', 'auto')
        try:
            xtl = np.asarray(xtl)[xind]
        except (TypeError, IndexError):
            pass
        ytl = kws.pop('yticklabels', 'auto')
        try:
            ytl = np.asarray(ytl)[yind]
        except (TypeError, IndexError):
            pass
        annot = kws.pop('annot', None)
        if annot is None or annot is False:
            pass
        else:
            if isinstance(annot, bool):
                annot_data = self.data2d
            else:
                annot_data = np.asarray(annot)
                if annot_data.shape != self.data2d.shape:
                    err = '`data` and `annot` must have same shape.'
                    raise ValueError(err)
                annot_data = annot_data[yind][:, xind]
            annot = annot_data
        kws.setdefault('cbar', self.ax_cbar is not None)
        heatmap(self.data2d, ax=self.ax_heatmap, cbar_ax=self.ax_cbar, cbar_kws=colorbar_kws, mask=self.mask, xticklabels=xtl, yticklabels=ytl, annot=annot, **kws)
        ytl = self.ax_heatmap.get_yticklabels()
        ytl_rot = None if not ytl else ytl[0].get_rotation()
        self.ax_heatmap.yaxis.set_ticks_position('right')
        self.ax_heatmap.yaxis.set_label_position('right')
        if ytl_rot is not None:
            ytl = self.ax_heatmap.get_yticklabels()
            plt.setp(ytl, rotation=ytl_rot)
        tight_params = dict(h_pad=0.02, w_pad=0.02)
        if self.ax_cbar is None:
            self._figure.tight_layout(**tight_params)
        else:
            self.ax_cbar.set_axis_off()
            self._figure.tight_layout(**tight_params)
            self.ax_cbar.set_axis_on()
            self.ax_cbar.set_position(self.cbar_pos)

    def plot(self, metric, method, colorbar_kws, row_cluster, col_cluster, row_linkage, col_linkage, tree_kws, **kws):
        if kws.get('square', False):
            msg = '``square=True`` ignored in clustermap'
            warnings.warn(msg)
            kws.pop('square')
        colorbar_kws = {} if colorbar_kws is None else colorbar_kws
        self.plot_dendrograms(row_cluster, col_cluster, metric, method, row_linkage=row_linkage, col_linkage=col_linkage, tree_kws=tree_kws)
        try:
            xind = self.dendrogram_col.reordered_ind
        except AttributeError:
            xind = np.arange(self.data2d.shape[1])
        try:
            yind = self.dendrogram_row.reordered_ind
        except AttributeError:
            yind = np.arange(self.data2d.shape[0])
        self.plot_colors(xind, yind, **kws)
        self.plot_matrix(colorbar_kws, xind, yind, **kws)
        return self