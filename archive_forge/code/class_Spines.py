from collections.abc import MutableMapping
import functools
import numpy as np
import matplotlib as mpl
from matplotlib import _api, _docstring
from matplotlib.artist import allow_rasterization
import matplotlib.transforms as mtransforms
import matplotlib.patches as mpatches
import matplotlib.path as mpath
class Spines(MutableMapping):
    """
    The container of all `.Spine`\\s in an Axes.

    The interface is dict-like mapping names (e.g. 'left') to `.Spine` objects.
    Additionally, it implements some pandas.Series-like features like accessing
    elements by attribute::

        spines['top'].set_visible(False)
        spines.top.set_visible(False)

    Multiple spines can be addressed simultaneously by passing a list::

        spines[['top', 'right']].set_visible(False)

    Use an open slice to address all spines::

        spines[:].set_visible(False)

    The latter two indexing methods will return a `SpinesProxy` that broadcasts all
    ``set_*()`` and ``set()`` calls to its members, but cannot be used for any other
    operation.
    """

    def __init__(self, **kwargs):
        self._dict = kwargs

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def __getstate__(self):
        return self._dict

    def __setstate__(self, state):
        self.__init__(**state)

    def __getattr__(self, name):
        try:
            return self._dict[name]
        except KeyError:
            raise AttributeError(f"'Spines' object does not contain a '{name}' spine")

    def __getitem__(self, key):
        if isinstance(key, list):
            unknown_keys = [k for k in key if k not in self._dict]
            if unknown_keys:
                raise KeyError(', '.join(unknown_keys))
            return SpinesProxy({k: v for k, v in self._dict.items() if k in key})
        if isinstance(key, tuple):
            raise ValueError('Multiple spines must be passed as a single list')
        if isinstance(key, slice):
            if key.start is None and key.stop is None and (key.step is None):
                return SpinesProxy(self._dict)
            else:
                raise ValueError('Spines does not support slicing except for the fully open slice [:] to access all spines.')
        return self._dict[key]

    def __setitem__(self, key, value):
        self._dict[key] = value

    def __delitem__(self, key):
        del self._dict[key]

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)