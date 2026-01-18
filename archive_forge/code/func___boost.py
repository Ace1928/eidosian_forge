import abc
import ctypes
import inspect
import json
import warnings
from collections import OrderedDict
from copy import deepcopy
from enum import Enum
from functools import wraps
from os import SEEK_END, environ
from os.path import getsize
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np
import scipy.sparse
from .compat import (PANDAS_INSTALLED, PYARROW_INSTALLED, arrow_cffi, arrow_is_floating, arrow_is_integer, concat,
from .libpath import find_lib_path
def __boost(self, grad: np.ndarray, hess: np.ndarray) -> bool:
    """Boost Booster for one iteration with customized gradient statistics.

        .. note::

            Score is returned before any transformation,
            e.g. it is raw margin instead of probability of positive class for binary task.
            For multi-class task, score are numpy 2-D array of shape = [n_samples, n_classes],
            and grad and hess should be returned in the same format.

        Parameters
        ----------
        grad : numpy 1-D array or numpy 2-D array (for multi-class task)
            The value of the first order derivative (gradient) of the loss
            with respect to the elements of score for each sample point.
        hess : numpy 1-D array or numpy 2-D array (for multi-class task)
            The value of the second order derivative (Hessian) of the loss
            with respect to the elements of score for each sample point.

        Returns
        -------
        is_finished : bool
            Whether the boost was successfully finished.
        """
    if self.__num_class > 1:
        grad = grad.ravel(order='F')
        hess = hess.ravel(order='F')
    grad = _list_to_1d_numpy(grad, dtype=np.float32, name='gradient')
    hess = _list_to_1d_numpy(hess, dtype=np.float32, name='hessian')
    assert grad.flags.c_contiguous
    assert hess.flags.c_contiguous
    if len(grad) != len(hess):
        raise ValueError(f"Lengths of gradient ({len(grad)}) and Hessian ({len(hess)}) don't match")
    num_train_data = self.train_set.num_data()
    if len(grad) != num_train_data * self.__num_class:
        raise ValueError(f"Lengths of gradient ({len(grad)}) and Hessian ({len(hess)}) don't match training data length ({num_train_data}) * number of models per one iteration ({self.__num_class})")
    is_finished = ctypes.c_int(0)
    _safe_call(_LIB.LGBM_BoosterUpdateOneIterCustom(self._handle, grad.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), hess.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), ctypes.byref(is_finished)))
    self.__is_predicted_cur_iter = [False for _ in range(self.__num_dataset)]
    return is_finished.value == 1