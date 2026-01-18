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
def _set_predictor(self, predictor: Optional[_InnerPredictor]) -> 'Dataset':
    """Set predictor for continued training.

        It is not recommended for user to call this function.
        Please use init_model argument in engine.train() or engine.cv() instead.
        """
    if predictor is None and self._predictor is None:
        return self
    elif isinstance(predictor, _InnerPredictor) and isinstance(self._predictor, _InnerPredictor):
        if predictor == self._predictor and predictor.current_iteration() == self._predictor.current_iteration():
            return self
    if self._handle is None:
        self._predictor = predictor
    elif self.data is not None:
        self._predictor = predictor
        self._set_init_score_by_predictor(predictor=self._predictor, data=self.data, used_indices=None)
    elif self.used_indices is not None and self.reference is not None and (self.reference.data is not None):
        self._predictor = predictor
        self._set_init_score_by_predictor(predictor=self._predictor, data=self.reference.data, used_indices=self.used_indices)
    else:
        raise LightGBMError('Cannot set predictor after freed raw data, set free_raw_data=False when construct Dataset to avoid this.')
    return self