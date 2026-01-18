from collections import OrderedDict, defaultdict
from itertools import cycle, tee
import bokeh.plotting as bkp
import matplotlib.pyplot as plt
import numpy as np
from bokeh.models import Band, ColumnDataSource, DataRange1d
from bokeh.models.annotations import Title, Legend
from bokeh.models.tickers import FixedTicker
from ....sel_utils import xarray_var_iter
from ....rcparams import rcParams
from ....stats import hdi
from ....stats.density_utils import get_bins, histogram, kde
from ....stats.diagnostics import _ess, _rhat
from ...plot_utils import _scale_fig_size
from .. import show_layout
from . import backend_kwarg_defaults
class PlotHandler:
    """Class to handle logic from ForestPlot."""

    def __init__(self, datasets, var_names, model_names, combined, combine_dims, colors, labeller):
        self.data = datasets
        if model_names is None:
            if len(self.data) > 1:
                model_names = [f'Model {idx}' for idx, _ in enumerate(self.data)]
            else:
                model_names = ['']
        elif len(model_names) != len(self.data):
            raise ValueError('The number of model names does not match the number of models')
        self.model_names = list(reversed(model_names))
        if var_names is None:
            if len(self.data) > 1:
                self.var_names = list(set().union(*[OrderedDict(datum.data_vars) for datum in self.data]))
            else:
                self.var_names = list(reversed(*[OrderedDict(datum.data_vars) for datum in self.data]))
        else:
            self.var_names = list(reversed(var_names))
        self.combined = combined
        self.combine_dims = combine_dims
        if colors == 'cycle':
            colors = [prop for _, prop in zip(range(len(self.data)), cycle(plt.rcParams['axes.prop_cycle'].by_key()['color']))]
        elif isinstance(colors, str):
            colors = [colors for _ in self.data]
        self.colors = list(reversed(colors))
        self.labeller = labeller
        self.plotters = self.make_plotters()

    def make_plotters(self):
        """Initialize an object for each variable to be plotted."""
        plotters, y = ({}, 0)
        for var_name in self.var_names:
            plotters[var_name] = VarHandler(var_name, self.data, y, model_names=self.model_names, combined=self.combined, combine_dims=self.combine_dims, colors=self.colors, labeller=self.labeller)
            y = plotters[var_name].y_max()
        return plotters

    def labels_and_ticks(self):
        """Collect labels and ticks from plotters."""
        val = self.plotters.values()

        def label_idxs():
            labels, idxs = ([], [])
            for plotter in val:
                sub_labels, sub_idxs, _, _, _ = plotter.labels_ticks_and_vals()
                labels_to_idxs = defaultdict(list)
                for label, idx in zip(sub_labels, sub_idxs):
                    labels_to_idxs[label].append(idx)
                sub_idxs = []
                sub_labels = []
                for label, all_idx in labels_to_idxs.items():
                    sub_labels.append(label)
                    sub_idxs.append(np.mean([j for j in all_idx]))
                labels.append(sub_labels)
                idxs.append(sub_idxs)
            return (np.concatenate(labels), np.concatenate(idxs))
        return label_idxs()

    def legend(self, ax, plotted):
        """Add interactive legend with colorcoded model info."""
        legend_it = []
        for model_name, glyphs in plotted.items():
            legend_it.append((model_name, glyphs))
        legend = Legend(items=legend_it, orientation='vertical', location='top_left')
        ax.add_layout(legend, 'above')
        ax.legend.click_policy = 'hide'

    def display_multiple_ropes(self, rope, ax, y, linewidth, var_name, selection, plotted, model_name):
        """Display ROPE when more than one interval is provided."""
        for sel in rope.get(var_name, []):
            if all((k in selection and selection[k] == v for k, v in sel.items() if k != 'rope')):
                vals = sel['rope']
                plotted[model_name].append(ax.line(vals, (y + 0.05, y + 0.05), line_width=linewidth * 2, color=[color for _, color in zip(range(3), cycle(plt.rcParams['axes.prop_cycle'].by_key()['color']))][2], line_alpha=0.7))
                return ax

    def ridgeplot(self, hdi_prob, mult, linewidth, markersize, alpha, ridgeplot_kind, ridgeplot_truncate, ridgeplot_quantiles, ax, plotted):
        """Draw ridgeplot for each plotter.

        Parameters
        ----------
        hdi_prob : float
            Probability for the highest density interval.
        mult : float
            How much to multiply height by. Set this to greater than 1 to have some overlap.
        linewidth : float
            Width of line on border of ridges
        markersize : float
            Size of marker in center of forestplot line
        alpha : float
            Transparency of ridges
        ridgeplot_kind : string
            By default ("auto") continuous variables are plotted using KDEs and discrete ones using
            histograms. To override this use "hist" to plot histograms and "density" for KDEs
        ridgeplot_truncate: bool
            Whether to truncate densities according to the value of hdi_prop. Defaults to True
        ridgeplot_quantiles: list
            Quantiles in ascending order used to segment the KDE. Use [.25, .5, .75] for quartiles.
            Defaults to None.
        ax : Axes
            Axes to draw on
        plotted : dict
            Contains glyphs for each model
        """
        if alpha is None:
            alpha = 1.0
        for plotter in list(self.plotters.values())[::-1]:
            for x, y_min, y_max, hdi_, y_q, color, model_name in plotter.ridgeplot(hdi_prob, mult, ridgeplot_kind):
                if alpha == 0:
                    border = color
                    facecolor = None
                else:
                    border = 'black'
                    facecolor = color
                if x.dtype.kind == 'i':
                    if ridgeplot_truncate:
                        y_max = y_max[(x >= hdi_[0]) & (x <= hdi_[1])]
                        x = x[(x >= hdi_[0]) & (x <= hdi_[1])]
                    else:
                        facecolor = color
                        alpha = [alpha if ci else 0 for ci in (x >= hdi_[0]) & (x <= hdi_[1])]
                    y_min = np.ones_like(x) * y_min
                    plotted[model_name].append(ax.vbar(x=x, top=y_max - y_min, bottom=y_min, width=0.9, line_color=border, color=facecolor, fill_alpha=alpha))
                else:
                    tr_x = x[(x >= hdi_[0]) & (x <= hdi_[1])]
                    tr_y_min = np.ones_like(tr_x) * y_min
                    tr_y_max = y_max[(x >= hdi_[0]) & (x <= hdi_[1])]
                    y_min = np.ones_like(x) * y_min
                    patch = ax.patch(np.concatenate([tr_x, tr_x[::-1]]), np.concatenate([tr_y_min, tr_y_max[::-1]]), fill_color=color, fill_alpha=alpha, line_width=0)
                    patch.level = 'overlay'
                    plotted[model_name].append(patch)
                    if ridgeplot_truncate:
                        plotted[model_name].append(ax.line(x, y_max, line_dash='solid', line_width=linewidth, line_color=border))
                        plotted[model_name].append(ax.line(x, y_min, line_dash='solid', line_width=linewidth, line_color=border))
                    else:
                        plotted[model_name].append(ax.line(tr_x, tr_y_max, line_dash='solid', line_width=linewidth, line_color=border))
                        plotted[model_name].append(ax.line(tr_x, tr_y_min, line_dash='solid', line_width=linewidth, line_color=border))
                if ridgeplot_quantiles is not None:
                    quantiles = [x[np.sum(y_q < quant)] for quant in ridgeplot_quantiles]
                    plotted[model_name].append(ax.diamond(quantiles, np.ones_like(quantiles) * y_min[0], line_color='black', fill_color='black', size=markersize))
        return ax

    def forestplot(self, hdi_prob, quartiles, linewidth, markersize, ax, rope, plotted):
        """Draw forestplot for each plotter.

        Parameters
        ----------
        hdi_prob : float
            Probability for the highest density interval. Width of each line.
        quartiles : bool
            Whether to mark quartiles
        linewidth : float
            Width of forestplot line
        markersize : float
            Size of marker in center of forestplot line
        ax : Axes
            Axes to draw on
        plotted : dict
            Contains glyphs for each model
        """
        if rope is None or isinstance(rope, dict):
            pass
        elif len(rope) == 2:
            cds = ColumnDataSource({'x': rope, 'lower': [-2 * self.y_max(), -2 * self.y_max()], 'upper': [self.y_max() * 2, self.y_max() * 2]})
            band = Band(base='x', lower='lower', upper='upper', fill_color=[color for _, color in zip(range(4), cycle(plt.rcParams['axes.prop_cycle'].by_key()['color']))][2], line_alpha=0.5, source=cds)
            ax.renderers.append(band)
        else:
            raise ValueError('Argument `rope` must be None, a dictionary like{"var_name": {"rope": (lo, hi)}}, or an iterable of length 2')
        endpoint = 100 * (1 - hdi_prob) / 2
        if quartiles:
            qlist = [endpoint, 25, 50, 75, 100 - endpoint]
        else:
            qlist = [endpoint, 50, 100 - endpoint]
        for plotter in self.plotters.values():
            for y, model_name, selection, values, color in plotter.treeplot(qlist, hdi_prob):
                if isinstance(rope, dict):
                    self.display_multiple_ropes(rope, ax, y, linewidth, plotter.var_name, selection, plotted, model_name)
                mid = len(values) // 2
                param_iter = zip(np.linspace(2 * linewidth, linewidth, mid, endpoint=True)[-1::-1], range(mid))
                for width, j in param_iter:
                    plotted[model_name].append(ax.line([values[j], values[-(j + 1)]], [y, y], line_width=width, line_color=color))
                plotted[model_name].append(ax.circle(x=values[mid], y=y, size=markersize * 0.75, fill_color=color))
        _title = Title()
        _title.text = f'{hdi_prob:.1%} HDI'
        ax.title = _title
        return ax

    def plot_neff(self, ax, markersize, plotted):
        """Draw effective n for each plotter."""
        max_ess = 0
        for plotter in self.plotters.values():
            for y, ess, color, model_name in plotter.ess():
                if ess is not None:
                    plotted[model_name].append(ax.circle(x=ess, y=y, fill_color=color, size=markersize, line_color='black'))
                if ess > max_ess:
                    max_ess = ess
        ax.x_range._property_values['start'] = 0
        ax.x_range._property_values['end'] = 1.07 * max_ess
        _title = Title()
        _title.text = 'ess'
        ax.title = _title
        ax.xaxis[0].ticker.desired_num_ticks = 3
        return ax

    def plot_rhat(self, ax, markersize, plotted):
        """Draw r-hat for each plotter."""
        for plotter in self.plotters.values():
            for y, r_hat, color, model_name in plotter.r_hat():
                if r_hat is not None:
                    plotted[model_name].append(ax.circle(x=r_hat, y=y, fill_color=color, size=markersize, line_color='black'))
        ax.x_range._property_values['start'] = 0.9
        ax.x_range._property_values['end'] = 2.1
        _title = Title()
        _title.text = 'r_hat'
        ax.title = _title
        ax.xaxis[0].ticker.desired_num_ticks = 3
        return ax

    def fig_height(self):
        """Figure out the height of this plot."""
        return 4 + len(self.data) * len(self.var_names) - 1 + 0.1 * sum((1 for j in self.plotters.values() for _ in j.iterator()))

    def y_max(self):
        """Get maximum y value for the plot."""
        return max((p.y_max() for p in self.plotters.values()))