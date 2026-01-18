from __future__ import annotations
import numpy as np
import pandas as pd
from pandas.core.resample import Resampler as pd_Resampler
from dask.base import tokenize
from dask.dataframe import methods
from dask.dataframe._compat import PANDAS_GE_140
from dask.dataframe.core import DataFrame, Series
from dask.highlevelgraph import HighLevelGraph
from dask.utils import derived_from
def _agg(self, how, meta=None, fill_value=np.nan, how_args=(), how_kwargs=None):
    """Aggregate using one or more operations

        Parameters
        ----------
        how : str
            Name of aggregation operation
        fill_value : scalar, optional
            Value to use for missing values, applied during upsampling.
            Default is NaN.
        how_args : optional
            Positional arguments for aggregation operation.
        how_kwargs : optional
            Keyword arguments for aggregation operation.

        Returns
        -------
        Dask DataFrame or Series
        """
    if how_kwargs is None:
        how_kwargs = {}
    rule = self._rule
    kwargs = self._kwargs
    name = 'resample-' + tokenize(self.obj, rule, kwargs, how, *how_args, **how_kwargs)
    newdivs, outdivs = _resample_bin_and_out_divs(self.obj.divisions, rule, **kwargs)
    partitioned = self.obj.repartition(newdivs, force=True)
    keys = partitioned.__dask_keys__()
    dsk = {}
    args = zip(keys, outdivs, outdivs[1:], ['left'] * (len(keys) - 1) + [None])
    for i, (k, s, e, c) in enumerate(args):
        dsk[name, i] = (_resample_series, k, s, e, c, rule, kwargs, how, fill_value, list(how_args), how_kwargs)
    meta_r = self.obj._meta_nonempty.resample(self._rule, **self._kwargs)
    meta = getattr(meta_r, how)(*how_args, **how_kwargs)
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=[partitioned])
    if isinstance(meta, pd.DataFrame):
        return DataFrame(graph, name, meta, outdivs)
    return Series(graph, name, meta, outdivs)