from typing import Any, Callable, Dict, List, Tuple, Union
import modin.pandas as pd
import numpy
import pandas
from triad.utils.assertion import assert_or_throw
from triad.utils.pandas_like import PandasLikeUtils
from qpd import QPDEngine, run_sql
from qpd.dataframe import Column, DataFrame
from qpd.specs import (
from qpd_pandas.engine import _RowsIndexer
class _LazyGrouper(RayUtils):

    def __init__(self, ndf: Any, gp_keys: List[str]):
        self.ndf = ndf
        self.gp_keys = gp_keys
        mi_data = [numpy.ndarray(0, ndf[x].dtype) for x in gp_keys]
        self.index = pandas.MultiIndex.from_arrays(mi_data, names=gp_keys)
        self.dtypes = {x: ndf[x].dtype for x in ndf.columns}

    def get(self, key: Union[None, str, List[str]], unique: bool, dropna: bool) -> Any:
        ndf = self.ndf
        if key is not None:
            if isinstance(key, str):
                ndf = self.ndf[self.gp_keys + [key]]
            else:
                ndf = self.ndf[self.gp_keys + key]
        if dropna:
            ndf = ndf.dropna(subset=[key])
        if unique:
            ndf = self.drop_duplicates(ndf)
        return ndf.groupby(self.gp_keys)