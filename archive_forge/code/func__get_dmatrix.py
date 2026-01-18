import logging
import math
import time
import warnings
from collections import defaultdict
from typing import Dict, List
import numpy as np
import pandas
import ray
import xgboost as xgb
from ray.util import get_node_ip_address
from modin.core.execution.ray.common import RayWrapper
from modin.distributed.dataframe.pandas import from_partitions
from .utils import RabitContext, RabitContextManager
def _get_dmatrix(self, X_y, **dmatrix_kwargs):
    """
        Create xgboost.DMatrix from sequence of pandas.DataFrame objects.

        First half of `X_y` should contains objects for `X`, second for `y`.

        Parameters
        ----------
        X_y : list
            List of pandas.DataFrame objects.
        **dmatrix_kwargs : dict
            Keyword parameters for ``xgb.DMatrix``.

        Returns
        -------
        xgb.DMatrix
            A XGBoost DMatrix.
        """
    s = time.time()
    X = X_y[:len(X_y) // 2]
    y = X_y[len(X_y) // 2:]
    assert len(X) == len(y) and len(X) > 0, 'X and y should have the equal length more than 0'
    X = pandas.concat(X, axis=0)
    y = pandas.concat(y, axis=0)
    LOGGER.info(f'Concat time: {time.time() - s} s')
    return xgb.DMatrix(X, y, nthread=self._nthreads, **dmatrix_kwargs)