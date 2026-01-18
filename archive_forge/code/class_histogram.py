import warnings
import numpy as np
import param
from packaging.version import Version
from param import _is_number
from ..core import (
from ..core.data import ArrayInterface, DictInterface, PandasInterface, default_datatype
from ..core.data.util import dask_array_module
from ..core.util import (
from ..element.chart import Histogram, Scatter
from ..element.path import Contours, Polygons
from ..element.raster import RGB, Image
from ..element.util import categorical_aggregate2d  # noqa (API import)
from ..streams import RangeXY
from ..util.locator import MaxNLocator
class histogram(Operation):
    """
    Returns a Histogram of the input element data, binned into
    num_bins over the bin_range (if specified) along the specified
    dimension.
    """
    bin_range = param.NumericTuple(default=None, length=2, doc='\n      Specifies the range within which to compute the bins.')
    bins = param.ClassSelector(default=None, class_=(np.ndarray, list, tuple, str), doc="\n      An explicit set of bin edges or a method to find the optimal\n      set of bin edges, e.g. 'auto', 'fd', 'scott' etc. For more\n      documentation on these approaches see the np.histogram_bin_edges\n      documentation.")
    cumulative = param.Boolean(default=False, doc='\n      Whether to compute the cumulative histogram')
    dimension = param.String(default=None, doc='\n      Along which dimension of the Element to compute the histogram.')
    frequency_label = param.String(default=None, doc='\n      Format string defining the label of the frequency dimension of the Histogram.')
    groupby = param.ClassSelector(default=None, class_=(str, Dimension), doc='\n      Defines a dimension to group the Histogram returning an NdOverlay of Histograms.')
    log = param.Boolean(default=False, doc='\n      Whether to use base 10 logarithmic samples for the bin edges.')
    mean_weighted = param.Boolean(default=False, doc='\n      Whether the weighted frequencies are averaged.')
    normed = param.ObjectSelector(default=False, objects=[True, False, 'integral', 'height'], doc="\n      Controls normalization behavior.  If `True` or `'integral'`, then\n      `density=True` is passed to np.histogram, and the distribution\n      is normalized such that the integral is unity.  If `False`,\n      then the frequencies will be raw counts. If `'height'`, then the\n      frequencies are normalized such that the max bin height is unity.")
    nonzero = param.Boolean(default=False, doc='\n      Whether to use only nonzero values when computing the histogram')
    num_bins = param.Integer(default=20, doc='\n      Number of bins in the histogram .')
    weight_dimension = param.String(default=None, doc='\n       Name of the dimension the weighting should be drawn from')
    style_prefix = param.String(default=None, allow_None=None, doc='\n      Used for setting a common style for histograms in a HoloMap or AdjointLayout.')

    def _process(self, element, key=None):
        if self.p.groupby:
            if not isinstance(element, Dataset):
                raise ValueError('Cannot use histogram groupby on non-Dataset Element')
            grouped = element.groupby(self.p.groupby, group_type=Dataset, container_type=NdOverlay)
            self.p.groupby = None
            return grouped.map(self._process, Dataset)
        normed = False if self.p.mean_weighted and self.p.weight_dimension else self.p.normed
        if self.p.dimension:
            selected_dim = self.p.dimension
        else:
            selected_dim = next((d.name for d in element.vdims + element.kdims))
        dim = element.get_dimension(selected_dim)
        if hasattr(element, 'interface'):
            data = element.interface.values(element, selected_dim, compute=False)
        else:
            data = element.dimension_values(selected_dim)
        is_datetime = isdatetime(data)
        if is_datetime:
            data = data.astype('datetime64[ns]').astype('int64')
        is_finite = isfinite
        is_cupy = is_cupy_array(data)
        if is_cupy:
            import cupy
            full_cupy_support = Version(cupy.__version__) > Version('8.0')
            if not full_cupy_support and (normed or self.p.weight_dimension):
                data = cupy.asnumpy(data)
                is_cupy = False
            else:
                is_finite = cupy.isfinite
        if is_ibis_expr(data):
            from ..core.data.ibis import ibis5
            mask = data.notnull()
            if self.p.nonzero:
                mask = mask & (data != 0)
            if ibis5():
                data = data.as_table()
            else:
                data = data.to_projection()
            data = data[mask]
            no_data = not len(data.head(1).execute())
            data = data[dim.name]
        else:
            mask = is_finite(data)
            if self.p.nonzero:
                mask = mask & (data != 0)
            data = data[mask]
            da = dask_array_module()
            no_data = False if da and isinstance(data, da.Array) else not len(data)
        if self.p.weight_dimension:
            if hasattr(element, 'interface'):
                weights = element.interface.values(element, self.p.weight_dimension, compute=False)
            else:
                weights = element.dimension_values(self.p.weight_dimension)
            weights = weights[mask]
        else:
            weights = None
        if isinstance(self.p.bins, str):
            bin_data = cupy.asnumpy(data) if is_cupy else data
            edges = np.histogram_bin_edges(bin_data, bins=self.p.bins)
        elif isinstance(self.p.bins, (list, np.ndarray)):
            edges = self.p.bins
            if isdatetime(edges):
                edges = edges.astype('datetime64[ns]').astype('int64')
        else:
            hist_range = self.p.bin_range or element.range(selected_dim)
            with warnings.catch_warnings():
                warnings.filterwarnings(action='ignore', message='elementwise comparison failed', category=DeprecationWarning)
                null_hist_range = hist_range == (0, 0)
            if null_hist_range or any((not isfinite(r) for r in hist_range)):
                hist_range = (0, 1)
            steps = self.p.num_bins + 1
            start, end = hist_range
            if isinstance(start, str) or isinstance(end, str) or isinstance(steps, str):
                raise ValueError('Categorical data found. Cannot create histogram from categorical data.')
            if is_datetime:
                start, end = (dt_to_int(start, 'ns'), dt_to_int(end, 'ns'))
            if self.p.log:
                bin_min = max([abs(start), data[data > 0].min()])
                edges = np.logspace(np.log10(bin_min), np.log10(end), steps)
            else:
                edges = np.linspace(start, end, steps)
        if is_cupy:
            edges = cupy.asarray(edges)
        if not is_dask_array(data) and no_data:
            nbins = self.p.num_bins if self.p.bins is None else len(self.p.bins) - 1
            hist = np.zeros(nbins)
        elif hasattr(element, 'interface'):
            density = True if normed else False
            hist, edges = element.interface.histogram(data, edges, density=density, weights=weights)
            if normed == 'height':
                hist /= hist.max()
            if self.p.weight_dimension and self.p.mean_weighted:
                hist_mean, _ = element.interface.histogram(data, density=False, bins=edges)
                hist /= hist_mean
        elif normed:
            hist, edges = np.histogram(data, density=True, weights=weights, bins=edges)
            if normed == 'height':
                hist /= hist.max()
        else:
            hist, edges = np.histogram(data, density=normed, weights=weights, bins=edges)
            if self.p.weight_dimension and self.p.mean_weighted:
                hist_mean, _ = np.histogram(data, density=False, bins=self.p.num_bins)
                hist /= hist_mean
        hist[np.isnan(hist)] = 0
        if is_datetime:
            edges = (edges / 1000.0).astype('datetime64[us]')
        params = {}
        if self.p.weight_dimension:
            params['vdims'] = [element.get_dimension(self.p.weight_dimension)]
        elif self.p.frequency_label:
            label = self.p.frequency_label.format(dim=dim.pprint_label)
            params['vdims'] = [Dimension('Frequency', label=label)]
        else:
            label = 'Frequency' if normed else 'Count'
            params['vdims'] = [Dimension(f'{dim.name}_{label.lower()}', label=label)]
        if element.group != element.__class__.__name__:
            params['group'] = element.group
        if self.p.cumulative:
            hist = np.cumsum(hist)
            if self.p.normed in (True, 'integral'):
                hist *= edges[1] - edges[0]
        self.bins = list(edges)
        return Histogram((edges, hist), kdims=[element.get_dimension(selected_dim)], label=element.label, **params)