import copy
from inspect import signature
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.sparse
from .basic import (Booster, Dataset, LightGBMError, _choose_param_value, _ConfigAliases, _LGBM_BoosterBestScoreType,
from .callback import _EvalResultDict, record_evaluation
from .compat import (SKLEARN_INSTALLED, LGBMNotFittedError, _LGBMAssertAllFinite, _LGBMCheckArray,
from .engine import train
class _EvalFunctionWrapper:
    """Proxy class for evaluation function."""

    def __init__(self, func: _LGBM_ScikitCustomEvalFunction):
        """Construct a proxy class.

        This class transforms evaluation function to match evaluation function with signature ``new_func(preds, dataset)``
        as expected by ``lightgbm.engine.train``.

        Parameters
        ----------
        func : callable
            Expects a callable with following signatures:
            ``func(y_true, y_pred)``,
            ``func(y_true, y_pred, weight)``
            or ``func(y_true, y_pred, weight, group)``
            and returns (eval_name, eval_result, is_higher_better) or
            list of (eval_name, eval_result, is_higher_better):

                y_true : numpy 1-D array of shape = [n_samples]
                    The target values.
                y_pred : numpy 1-D array of shape = [n_samples] or numpy 2-D array shape = [n_samples, n_classes] (for multi-class task)
                    The predicted values.
                    In case of custom ``objective``, predicted values are returned before any transformation,
                    e.g. they are raw margin instead of probability of positive class for binary task in this case.
                weight : numpy 1-D array of shape = [n_samples]
                    The weight of samples. Weights should be non-negative.
                group : numpy 1-D array
                    Group/query data.
                    Only used in the learning-to-rank task.
                    sum(group) = n_samples.
                    For example, if you have a 100-document dataset with ``group = [10, 20, 40, 10, 10, 10]``, that means that you have 6 groups,
                    where the first 10 records are in the first group, records 11-30 are in the second group, records 31-70 are in the third group, etc.
                eval_name : str
                    The name of evaluation function (without whitespace).
                eval_result : float
                    The eval result.
                is_higher_better : bool
                    Is eval result higher better, e.g. AUC is ``is_higher_better``.
        """
        self.func = func

    def __call__(self, preds: np.ndarray, dataset: Dataset) -> Union[_LGBM_EvalFunctionResultType, List[_LGBM_EvalFunctionResultType]]:
        """Call passed function with appropriate arguments.

        Parameters
        ----------
        preds : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)
            The predicted values.
        dataset : Dataset
            The training dataset.

        Returns
        -------
        eval_name : str
            The name of evaluation function (without whitespace).
        eval_result : float
            The eval result.
        is_higher_better : bool
            Is eval result higher better, e.g. AUC is ``is_higher_better``.
        """
        labels = _get_label_from_constructed_dataset(dataset)
        argc = len(signature(self.func).parameters)
        if argc == 2:
            return self.func(labels, preds)
        weight = _get_weight_from_constructed_dataset(dataset)
        if argc == 3:
            return self.func(labels, preds, weight)
        if argc == 4:
            group = _get_group_from_constructed_dataset(dataset)
            return self.func(labels, preds, weight, group)
        raise TypeError(f'Self-defined eval function should have 2, 3 or 4 arguments, got {argc}')