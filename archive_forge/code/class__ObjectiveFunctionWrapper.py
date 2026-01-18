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
class _ObjectiveFunctionWrapper:
    """Proxy class for objective function."""

    def __init__(self, func: _LGBM_ScikitCustomObjectiveFunction):
        """Construct a proxy class.

        This class transforms objective function to match objective function with signature ``new_func(preds, dataset)``
        as expected by ``lightgbm.engine.train``.

        Parameters
        ----------
        func : callable
            Expects a callable with following signatures:
            ``func(y_true, y_pred)``,
            ``func(y_true, y_pred, weight)``
            or ``func(y_true, y_pred, weight, group)``
            and returns (grad, hess):

                y_true : numpy 1-D array of shape = [n_samples]
                    The target values.
                y_pred : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)
                    The predicted values.
                    Predicted values are returned before any transformation,
                    e.g. they are raw margin instead of probability of positive class for binary task.
                weight : numpy 1-D array of shape = [n_samples]
                    The weight of samples. Weights should be non-negative.
                group : numpy 1-D array
                    Group/query data.
                    Only used in the learning-to-rank task.
                    sum(group) = n_samples.
                    For example, if you have a 100-document dataset with ``group = [10, 20, 40, 10, 10, 10]``, that means that you have 6 groups,
                    where the first 10 records are in the first group, records 11-30 are in the second group, records 31-70 are in the third group, etc.
                grad : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape [n_samples, n_classes] (for multi-class task)
                    The value of the first order derivative (gradient) of the loss
                    with respect to the elements of y_pred for each sample point.
                hess : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)
                    The value of the second order derivative (Hessian) of the loss
                    with respect to the elements of y_pred for each sample point.

        .. note::

            For multi-class task, y_pred is a numpy 2-D array of shape = [n_samples, n_classes],
            and grad and hess should be returned in the same format.
        """
        self.func = func

    def __call__(self, preds: np.ndarray, dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """Call passed function with appropriate arguments.

        Parameters
        ----------
        preds : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)
            The predicted values.
        dataset : Dataset
            The training dataset.

        Returns
        -------
        grad : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)
            The value of the first order derivative (gradient) of the loss
            with respect to the elements of preds for each sample point.
        hess : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)
            The value of the second order derivative (Hessian) of the loss
            with respect to the elements of preds for each sample point.
        """
        labels = _get_label_from_constructed_dataset(dataset)
        argc = len(signature(self.func).parameters)
        if argc == 2:
            grad, hess = self.func(labels, preds)
            return (grad, hess)
        weight = _get_weight_from_constructed_dataset(dataset)
        if argc == 3:
            grad, hess = self.func(labels, preds, weight)
            return (grad, hess)
        if argc == 4:
            group = _get_group_from_constructed_dataset(dataset)
            return self.func(labels, preds, weight, group)
        raise TypeError(f'Self-defined objective function should have 2, 3 or 4 arguments, got {argc}')