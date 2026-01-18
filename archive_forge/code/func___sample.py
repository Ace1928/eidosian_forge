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
def __sample(self, seqs: List[Sequence], total_nrow: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Sample data from seqs.

        Mimics behavior in c_api.cpp:LGBM_DatasetCreateFromMats()

        Returns
        -------
            sampled_rows, sampled_row_indices
        """
    indices = self._create_sample_indices(total_nrow)
    sampled = np.array(list(self._yield_row_from_seqlist(seqs, indices)))
    sampled = sampled.T
    filtered = []
    filtered_idx = []
    sampled_row_range = np.arange(len(indices), dtype=np.int32)
    for col in sampled:
        col_predicate = (np.abs(col) > ZERO_THRESHOLD) | np.isnan(col)
        filtered_col = col[col_predicate]
        filtered_row_idx = sampled_row_range[col_predicate]
        filtered.append(filtered_col)
        filtered_idx.append(filtered_row_idx)
    return (filtered, filtered_idx)