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
class VarHandler:
    """Handle individual variable logic."""

    def __init__(self, var_name, data, y_start, model_names, combined, combine_dims, colors, labeller):
        self.var_name = var_name
        self.data = data
        self.y_start = y_start
        self.model_names = model_names
        self.combined = combined
        self.combine_dims = combine_dims
        self.colors = colors
        self.labeller = labeller
        self.model_color = dict(zip(self.model_names, self.colors))
        max_chains = max((datum.chain.max().values for datum in data))
        self.chain_offset = len(data) * 0.45 / max(1, max_chains)
        self.var_offset = 1.5 * self.chain_offset
        self.group_offset = 2 * self.var_offset

    def iterator(self):
        """Iterate over models and chains for each variable."""
        if self.combined:
            grouped_data = [[(0, datum)] for datum in self.data]
            skip_dims = self.combine_dims.union({'chain'})
        else:
            grouped_data = [datum.groupby('chain', squeeze=False) for datum in self.data]
            skip_dims = self.combine_dims
        label_dict = OrderedDict()
        selection_list = []
        for name, grouped_datum in zip(self.model_names, grouped_data):
            for _, sub_data in grouped_datum:
                datum_iter = xarray_var_iter(sub_data.squeeze(), var_names=[self.var_name], skip_dims=skip_dims, reverse_selections=True)
                datum_list = list(datum_iter)
                for _, selection, isel, values in datum_list:
                    selection_list.append(selection)
                    if not selection or not len(selection_list) % len(datum_list):
                        var_name = self.var_name
                    else:
                        var_name = ''
                    label = self.labeller.make_label_flat(var_name, selection, isel)
                    if label not in label_dict:
                        label_dict[label] = OrderedDict()
                    if name not in label_dict[label]:
                        label_dict[label][name] = []
                    label_dict[label][name].append(values)
        y = self.y_start
        for idx, (label, model_data) in enumerate(label_dict.items()):
            for model_name, value_list in model_data.items():
                row_label = self.labeller.make_model_label(model_name, label)
                for values in value_list:
                    yield (y, row_label, model_name, label, selection_list[idx], values, self.model_color[model_name])
                    y += self.chain_offset
                y += self.var_offset
            y += self.group_offset

    def labels_ticks_and_vals(self):
        """Get labels, ticks, values, and colors for the variable."""
        y_ticks = defaultdict(list)
        for y, label, model_name, _, _, vals, color in self.iterator():
            y_ticks[label].append((y, vals, color, model_name))
        labels, ticks, vals, colors, model_names = ([], [], [], [], [])
        for label, all_data in y_ticks.items():
            for data in all_data:
                labels.append(label)
                ticks.append(data[0])
                vals.append(np.array(data[1]))
                model_names.append(data[3])
                colors.append(data[2])
        return (labels, ticks, vals, colors, model_names)

    def treeplot(self, qlist, hdi_prob):
        """Get data for each treeplot for the variable."""
        for y, _, model_name, _, selection, values, color in self.iterator():
            ntiles = np.percentile(values.flatten(), qlist)
            ntiles[0], ntiles[-1] = hdi(values.flatten(), hdi_prob, multimodal=False)
            yield (y, model_name, selection, ntiles, color)

    def ridgeplot(self, hdi_prob, mult, ridgeplot_kind):
        """Get data for each ridgeplot for the variable."""
        xvals, hdi_vals, yvals, pdfs, pdfs_q, colors, model_names = ([], [], [], [], [], [], [])
        for y, _, model_name, *_, values, color in self.iterator():
            yvals.append(y)
            colors.append(color)
            model_names.append(model_name)
            values = values.flatten()
            values = values[np.isfinite(values)]
            if hdi_prob != 1:
                hdi_ = hdi(values, hdi_prob, multimodal=False)
            else:
                hdi_ = (min(values), max(values))
            if ridgeplot_kind == 'auto':
                kind = 'hist' if np.all(np.mod(values, 1) == 0) else 'density'
            else:
                kind = ridgeplot_kind
            if kind == 'hist':
                bins = get_bins(values)
                _, density, x = histogram(values, bins=bins)
                x = x[:-1]
            elif kind == 'density':
                x, density = kde(values)
            density_q = density.cumsum() / density.sum()
            xvals.append(x)
            pdfs.append(density)
            pdfs_q.append(density_q)
            hdi_vals.append(hdi_)
        scaling = max((np.max(j) for j in pdfs))
        for y, x, hdi_val, pdf, pdf_q, color, model_name in zip(yvals, xvals, hdi_vals, pdfs, pdfs_q, colors, model_names):
            yield (x, y, mult * pdf / scaling + y, hdi_val, pdf_q, color, model_name)

    def ess(self):
        """Get effective n data for the variable."""
        _, y_vals, values, colors, model_names = self.labels_ticks_and_vals()
        for y, value, color, model_name in zip(y_vals, values, colors, model_names):
            yield (y, _ess(value), color, model_name)

    def r_hat(self):
        """Get rhat data for the variable."""
        _, y_vals, values, colors, model_names = self.labels_ticks_and_vals()
        for y, value, color, model_name in zip(y_vals, values, colors, model_names):
            if value.ndim != 2 or value.shape[0] < 2:
                yield (y, None, color, model_name)
            else:
                yield (y, _rhat(value), color, model_name)

    def y_max(self):
        """Get max y value for the variable."""
        end_y = max((y for y, *_ in self.iterator()))
        if self.combined:
            end_y += self.group_offset
        return end_y + 2 * self.group_offset