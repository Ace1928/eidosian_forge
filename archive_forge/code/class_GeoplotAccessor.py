import warnings
import numpy as np
import pandas as pd
from pandas.plotting import PlotAccessor
from pandas import CategoricalDtype
import geopandas
from packaging.version import Version
from ._decorator import doc
@doc(plot_dataframe)
class GeoplotAccessor(PlotAccessor):
    _pandas_kinds = PlotAccessor._all_kinds

    def __call__(self, *args, **kwargs):
        data = self._parent.copy()
        kind = kwargs.pop('kind', 'geo')
        if kind == 'geo':
            return plot_dataframe(data, *args, **kwargs)
        if kind in self._pandas_kinds:
            return PlotAccessor(data)(kind=kind, **kwargs)
        else:
            raise ValueError(f'{kind} is not a valid plot kind')

    def geo(self, *args, **kwargs):
        return self(*args, kind='geo', **kwargs)