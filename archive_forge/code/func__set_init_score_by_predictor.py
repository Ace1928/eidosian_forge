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
def _set_init_score_by_predictor(self, predictor: Optional[_InnerPredictor], data: _LGBM_TrainDataType, used_indices: Optional[Union[List[int], np.ndarray]]) -> 'Dataset':
    data_has_header = False
    if isinstance(data, (str, Path)) and self.params is not None:
        data_has_header = any((self.params.get(alias, False) for alias in _ConfigAliases.get('header')))
    num_data = self.num_data()
    if predictor is not None:
        init_score: Union[np.ndarray, scipy.sparse.spmatrix] = predictor.predict(data=data, raw_score=True, data_has_header=data_has_header)
        init_score = init_score.ravel()
        if used_indices is not None:
            assert not self._need_slice
            if isinstance(data, (str, Path)):
                sub_init_score = np.empty(num_data * predictor.num_class, dtype=np.float64)
                assert num_data == len(used_indices)
                for i in range(len(used_indices)):
                    for j in range(predictor.num_class):
                        sub_init_score[i * predictor.num_class + j] = init_score[used_indices[i] * predictor.num_class + j]
                init_score = sub_init_score
        if predictor.num_class > 1:
            new_init_score = np.empty(init_score.size, dtype=np.float64)
            for i in range(num_data):
                for j in range(predictor.num_class):
                    new_init_score[j * num_data + i] = init_score[i * predictor.num_class + j]
            init_score = new_init_score
    elif self.init_score is not None:
        init_score = np.full_like(self.init_score, fill_value=0.0, dtype=np.float64)
    else:
        return self
    self.set_init_score(init_score)
    return self