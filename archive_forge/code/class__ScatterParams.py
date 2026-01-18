from .. import select
from .. import utils
from .._lazyload import matplotlib as mpl
from . import colors
from .tools import create_colormap
from .tools import create_normalize
from .tools import generate_colorbar
from .tools import generate_legend
from .tools import label_axis
from .utils import _get_figure
from .utils import _in_ipynb
from .utils import _is_color_array
from .utils import _with_default
from .utils import parse_fontsize
from .utils import show
from .utils import temp_fontsize
import numbers
import numpy as np
import pandas as pd
import warnings
class _ScatterParams(object):

    def __init__(self, x, y, z=None, c=None, mask=None, discrete=None, cmap=None, cmap_scale=None, vmin=None, vmax=None, s=None, legend=None, colorbar=None, xlabel=None, ylabel=None, zlabel=None, label_prefix=None, shuffle=True):
        self._x = x
        self._y = y
        self._z = z if z is not None else None
        self._c = c
        self._mask = mask
        self._discrete = discrete
        self._cmap = cmap
        self._cmap_scale = cmap_scale
        self._vmin_set = vmin
        self._vmax_set = vmax
        self._s = s
        self._legend = legend
        self._colorbar = colorbar
        self._labels = None
        self._c_discrete = None
        self._label_prefix = label_prefix
        self._xlabel = xlabel
        self._ylabel = ylabel
        self._zlabel = zlabel
        self.shuffle = shuffle
        self.check_size()
        self.check_c()
        self.check_mask()
        self.check_s()
        self.check_discrete()
        self.check_legend()
        self.check_cmap()
        self.check_cmap_scale()
        self.check_vmin_vmax()

    @property
    def x_array(self):
        return _squeeze_array(self._x)

    @property
    def y_array(self):
        return _squeeze_array(self._y)

    @property
    def z_array(self):
        return _squeeze_array(self._z) if self._z is not None else None

    @property
    def size(self):
        try:
            return self._size
        except AttributeError:
            self._size = len(self.x_array)
            return self._size

    @property
    def plot_idx(self):
        try:
            return self._plot_idx
        except AttributeError:
            self._plot_idx = np.arange(self.size)
            if self._mask is not None:
                self._plot_idx = self._plot_idx[self._mask]
            if self.shuffle:
                self._plot_idx = np.random.permutation(self._plot_idx)
            return self._plot_idx

    @property
    def x(self):
        return self.x_array[self.plot_idx]

    @property
    def y(self):
        return self.y_array[self.plot_idx]

    @property
    def z(self):
        return self.z_array[self.plot_idx] if self._z is not None else None

    @property
    def data(self):
        if self.z is not None:
            return [self.x, self.y, self.z]
        else:
            return [self.x, self.y]

    @property
    def _data(self):
        if self._z is not None:
            return [self.x_array, self.y_array, self.z_array]
        else:
            return [self.x_array, self.y_array]

    @property
    def s(self):
        if self._s is not None:
            if isinstance(self._s, numbers.Number):
                return self._s
            else:
                return self._s[self.plot_idx]
        else:
            return 200 / np.sqrt(self.size)

    def constant_c(self):
        """Check if ``c`` is constant.

        Returns
        -------
        c : ``str`` or ``None``
            Either None or a single matplotlib color
        """
        if self._c is None or isinstance(self._c, str):
            return True
        elif hasattr(self._c, '__len__') and len(self._c) == self.size:
            return False
        else:
            return mpl.colors.is_color_like(self._c)

    def array_c(self):
        """Check if ``c`` is an array of matplotlib colors."""
        try:
            return self._array_c
        except AttributeError:
            self._array_c = not self.constant_c() and _is_color_array(self._c)
            return self._array_c

    @property
    def _c_masked(self):
        if self.constant_c() or self._mask is None:
            return self._c
        else:
            return self._c[self._mask]

    @property
    def c_unique(self):
        """Get unique values in c to avoid recomputing every time."""
        try:
            return self._c_unique
        except AttributeError:
            self._c_unique = np.unique(self._c_masked)
            return self._c_unique

    @property
    def n_c_unique(self):
        """Get the number of unique values in `c`."""
        try:
            return self._n_c_unique
        except AttributeError:
            self._n_c_unique = len(self.c_unique)
            return self._n_c_unique

    @property
    def discrete(self):
        """Check if the color array is discrete.

        If not provided:
        * If c is constant or an array, return None
        * If cmap is a dict, return True
        * If c has 20 or less unique values, return True
        * Otherwise, return False
        """
        if self._discrete is not None:
            return self._discrete
        elif self.constant_c() or self.array_c():
            return None
        elif isinstance(self._cmap, dict) or not np.all([isinstance(x, numbers.Number) for x in self._c_masked]):
            return True
        elif self.n_c_unique > 20:
            return False
        else:
            return np.allclose(self.c_unique % 1, 0, atol=0.0001)

    @property
    def c_discrete(self):
        """Discretize ``c``.

        If c is discrete then this converts it to
        integers from 0 to `n_c_unique`
        """
        if self._c_discrete is None:
            if isinstance(self._cmap, dict):
                self._labels = np.array([k for k in self._cmap.keys() if k in self.c_unique])
                self._c_discrete = np.zeros_like(self._c, dtype=int)
                for i, label in enumerate(self._labels):
                    self._c_discrete[self._c == label] = i
            else:
                self._c_discrete = np.zeros_like(self._c, dtype=int)
                self._c_discrete[self._mask], self._labels = pd.factorize(self._c_masked, sort=True)
        return self._c_discrete

    @property
    def c(self):
        if self.constant_c():
            return self._c
        elif self.array_c() or not self.discrete:
            return self._c[self.plot_idx]
        else:
            return self.c_discrete[self.plot_idx]

    @property
    def labels(self):
        """Get labels associated with each integer c, if c is discrete."""
        if self.constant_c() or self.array_c():
            return None
        elif self.discrete:
            self.c_discrete
            return self._labels
        else:
            return None

    @property
    def legend(self):
        if self._legend is not None:
            return self._legend
        elif self.constant_c() or self.array_c():
            return False
        else:
            return True

    def list_cmap(self):
        """Check if the colormap is a list."""
        return hasattr(self._cmap, '__len__') and (not isinstance(self._cmap, (str, dict)))

    def process_string_cmap(self, cmap):
        """Subset a discrete colormap based on the number of colors if necessary."""
        cmap = mpl.cm.get_cmap(cmap)
        if self.discrete and cmap.N <= 20 and (self.n_c_unique <= cmap.N):
            return mpl.colors.ListedColormap(cmap.colors[:self.n_c_unique])
        else:
            return cmap

    @property
    def cmap(self):
        if self._cmap is not None:
            if isinstance(self._cmap, dict):
                return mpl.colors.ListedColormap([mpl.colors.to_rgba(self._cmap[label]) for label in self.labels])
            elif self.list_cmap():
                return create_colormap(self._cmap)
            elif isinstance(self._cmap, str):
                return self.process_string_cmap(self._cmap)
            else:
                return self._cmap
        elif self.constant_c() or self.array_c():
            return None
        elif self.discrete:
            return colors.tab(n=self.n_c_unique)
        else:
            return self.process_string_cmap('inferno')

    @property
    def cmap_scale(self):
        if self._cmap_scale is not None:
            return self._cmap_scale
        elif self.discrete or not self.legend:
            return None
        else:
            return 'linear'

    @property
    def _use_norm(self):
        return self.cmap_scale is not None and self.cmap_scale != 'linear'

    @property
    def _vmin(self):
        if self._vmin_set is not None:
            return self._vmin_set
        elif self.constant_c() or self.array_c() or self.discrete:
            return None
        else:
            return np.nanmin(self.c)

    @property
    def vmin(self):
        if self._use_norm:
            return None
        else:
            return self._vmin

    @property
    def _vmax(self):
        if self._vmax_set is not None:
            return self._vmax_set
        elif self.constant_c() or self.array_c() or self.discrete:
            return None
        else:
            return np.nanmax(self.c)

    @property
    def vmax(self):
        if self._use_norm:
            return None
        else:
            return self._vmax

    @property
    def norm(self):
        if self._use_norm:
            return create_normalize(self._vmin, self._vmax, scale=self.cmap_scale)
        else:
            return None

    @property
    def extend(self):
        if self.legend and (not self.discrete):
            extend_min = np.min(self.c) < self._vmin
            extend_max = np.max(self.c) > self._vmax
            if extend_min:
                return 'both' if extend_max else 'min'
            else:
                return 'max' if extend_max else 'neither'
        else:
            return None

    @property
    def subplot_kw(self):
        if self.z is not None:
            return {'projection': '3d'}
        else:
            return {}

    def check_vmin_vmax(self):
        if self.constant_c():
            if self._vmin_set is not None or self._vmax_set is not None:
                warnings.warn('Cannot set `vmin` or `vmax` with constant `c={}`. Setting `vmin = vmax = None`.'.format(self.c), UserWarning)
            self._vmin_set = None
            self._vmax_set = None
        elif self.discrete:
            if self._vmin_set is not None or self._vmax_set is not None:
                warnings.warn('Cannot set `vmin` or `vmax` with discrete data. Setting to `None`.', UserWarning)
            self._vmin_set = None
            self._vmax_set = None

    def check_legend(self):
        if self._colorbar is not None:
            if self._legend is not None and self._legend != self._colorbar:
                raise ValueError('Received conflicting values for synonyms `legend={}` and `colorbar={}`'.format(self._legend, self._colorbar))
            else:
                self._legend = self._colorbar
        if self._legend:
            if self.array_c():
                warnings.warn('`c` is a color array and cannot be used to create a legend. To interpret these values as labels instead, provide a `cmap` dictionary with label-color pairs.', UserWarning)
                self._legend = False
            elif self.constant_c():
                warnings.warn('Cannot create a legend with constant `c={}`'.format(self.c), UserWarning)
                self._legend = False

    def check_size(self):
        for d in self._data:
            if len(d) != self.size:
                raise ValueError('Expected all axes of data to have the same length. Got {}'.format([len(d) for d in self._data]))

    def check_c(self):
        if not self.constant_c():
            self._c = _squeeze_array(self._c)
            if not len(self._c) == self.size:
                raise ValueError('Expected c of length {} or 1. Got {}'.format(self.size, len(self._c)))

    def check_mask(self):
        if self._mask is not None:
            self._mask = _squeeze_array(self._mask)
            if not len(self._mask) == self.size:
                raise ValueError('Expected mask of length {}. Got {}'.format(self.size, len(self._mask)))

    def check_s(self):
        if self._s is not None and (not isinstance(self._s, numbers.Number)):
            self._s = _squeeze_array(self._s)
            if not len(self._s) == self.size:
                raise ValueError('Expected s of length {} or 1. Got {}'.format(self.size, len(self._s)))

    def check_discrete(self):
        if self._discrete is False:
            if not np.all([isinstance(x, numbers.Number) for x in self._c]):
                raise ValueError('Cannot treat non-numeric data as continuous.')

    def check_cmap(self):
        if isinstance(self._cmap, dict):
            if self.constant_c() or self.array_c():
                raise ValueError('Expected list-like `c` with dictionary cmap. Got {}'.format(type(self._c)))
            elif not self.discrete:
                raise ValueError('Cannot use dictionary cmap with continuous data.')
            elif np.any([color not in self._cmap for color in np.unique(self._c)]):
                missing = set(np.unique(self._c).tolist()).difference(self._cmap.keys())
                raise ValueError('Dictionary cmap requires a color for every unique entry in `c`. Missing colors for [{}]'.format(', '.join([str(color) for color in missing])))
        elif self.list_cmap():
            if self.constant_c() or self.array_c():
                raise ValueError('Expected list-like `c` with list cmap. Got {}'.format(type(self._c)))

    def check_cmap_scale(self):
        if self._cmap_scale is not None and self._cmap_scale != 'linear':
            if self.array_c():
                warnings.warn('Cannot use non-linear `cmap_scale` with `c` as a color array.', UserWarning)
                self._cmap_scale = 'linear'
            elif self.constant_c():
                warnings.warn('Cannot use non-linear `cmap_scale` with constant `c={}`.'.format(self._c), UserWarning)
                self._cmap_scale = 'linear'
            elif self.discrete:
                warnings.warn('Cannot use non-linear `cmap_scale` with discrete data.', UserWarning)
                self._cmap_scale = 'linear'

    def _label(self, label, values, idx):
        if label is False:
            return None
        elif label is not None:
            return label
        elif self._label_prefix is not None:
            return self._label_prefix + str(idx)
        elif label is not False and isinstance(values, pd.Series):
            return values.name
        else:
            return None

    @property
    def xlabel(self):
        return self._label(self._xlabel, self._x, '1')

    @property
    def ylabel(self):
        return self._label(self._ylabel, self._y, '2')

    @property
    def zlabel(self):
        if self._z is None:
            return None
        else:
            return self._label(self._zlabel, self._z, '3')