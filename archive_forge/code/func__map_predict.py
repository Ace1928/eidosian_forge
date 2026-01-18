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
@ray.remote
def _map_predict(booster, part, columns, dmatrix_kwargs={}, **kwargs):
    """
    Run prediction on a remote worker.

    Parameters
    ----------
    booster : xgboost.Booster or ray.ObjectRef
        A trained booster.
    part : pandas.DataFrame or ray.ObjectRef
        Partition of full data used for local prediction.
    columns : list or ray.ObjectRef
        Columns for the result.
    dmatrix_kwargs : dict, optional
        Keyword parameters for ``xgb.DMatrix``.
    **kwargs : dict
        Other parameters are the same as for ``xgboost.Booster.predict``.

    Returns
    -------
    ray.ObjectRef
        ``ray.ObjectRef`` with partial prediction.
    """
    dmatrix = xgb.DMatrix(part, **dmatrix_kwargs)
    prediction = pandas.DataFrame(booster.predict(dmatrix, **kwargs), index=part.index, columns=columns)
    return prediction