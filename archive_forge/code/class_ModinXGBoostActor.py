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
@ray.remote(num_cpus=0)
class ModinXGBoostActor:
    """
    Ray actor-class runs training on the remote worker.

    Parameters
    ----------
    rank : int
        Rank of this actor.
    nthread : int
        Number of threads used by XGBoost in this actor.
    """

    def __init__(self, rank, nthread):
        self._evals = []
        self._rank = rank
        self._nthreads = nthread
        LOGGER.info(f'Actor <{self._rank}>, nthread = {self._nthreads} was initialized.')

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

    def set_train_data(self, *X_y, add_as_eval_method=None, **dmatrix_kwargs):
        """
        Set train data for actor.

        Parameters
        ----------
        *X_y : iterable
            Sequence of ray.ObjectRef objects. First half of sequence is for
            `X` data, second for `y`. When it is passed in actor, auto-materialization
            of ray.ObjectRef -> pandas.DataFrame happens.
        add_as_eval_method : str, optional
            Name of eval data. Used in case when train data also used for evaluation.
        **dmatrix_kwargs : dict
            Keyword parameters for ``xgb.DMatrix``.
        """
        self._dtrain = self._get_dmatrix(X_y, **dmatrix_kwargs)
        if add_as_eval_method is not None:
            self._evals.append((self._dtrain, add_as_eval_method))

    def add_eval_data(self, *X_y, eval_method, **dmatrix_kwargs):
        """
        Add evaluation data for actor.

        Parameters
        ----------
        *X_y : iterable
            Sequence of ray.ObjectRef objects. First half of sequence is for
            `X` data, second for `y`. When it is passed in actor, auto-materialization
            of ray.ObjectRef -> pandas.DataFrame happens.
        eval_method : str
            Name of eval data.
        **dmatrix_kwargs : dict
            Keyword parameters for ``xgb.DMatrix``.
        """
        self._evals.append((self._get_dmatrix(X_y, **dmatrix_kwargs), eval_method))

    def train(self, rabit_args, params, *args, **kwargs):
        """
        Run local XGBoost training.

        Connects to Rabit Tracker environment to share training data between
        actors and trains XGBoost booster using `self._dtrain`.

        Parameters
        ----------
        rabit_args : list
            List with environment variables for Rabit Tracker.
        params : dict
            Booster params.
        *args : iterable
            Other parameters for `xgboost.train`.
        **kwargs : dict
            Other parameters for `xgboost.train`.

        Returns
        -------
        dict
            A dictionary with trained booster and dict of
            evaluation results
            as {"booster": xgb.Booster, "history": dict}.
        """
        local_params = params.copy()
        local_dtrain = self._dtrain
        local_evals = self._evals
        local_params['nthread'] = self._nthreads
        evals_result = dict()
        s = time.time()
        with RabitContext(self._rank, rabit_args):
            bst = xgb.train(local_params, local_dtrain, *args, evals=local_evals, evals_result=evals_result, **kwargs)
            LOGGER.info(f'Local training time: {time.time() - s} s')
            return {'booster': bst, 'history': evals_result}