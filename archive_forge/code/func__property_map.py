from __future__ import annotations
import functools
import warnings
import numpy as np
import pandas as pd
from dask.dataframe._compat import check_to_pydatetime_deprecation
from dask.utils import derived_from
def _property_map(self, attr):
    meta = self._delegate_property(self._series._meta, self._accessor_name, attr)
    token = f'{self._accessor_name}-{attr}'
    return self._series.map_partitions(self._delegate_property, self._accessor_name, attr, token=token, meta=meta)