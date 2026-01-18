import gc
import importlib.util
import multiprocessing
import os
import platform
import socket
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from io import StringIO
from platform import system
from typing import (
import numpy as np
import pytest
from scipy import sparse
import xgboost as xgb
from xgboost.core import ArrayLike
from xgboost.sklearn import SklObjective
from xgboost.testing.data import (
from hypothesis import strategies
from hypothesis.extra.numpy import arrays
class IteratorForTest(xgb.core.DataIter):
    """Iterator for testing streaming DMatrix. (external memory, quantile)"""

    def __init__(self, X: Sequence, y: Sequence, w: Optional[Sequence], cache: Optional[str]) -> None:
        assert len(X) == len(y)
        self.X = X
        self.y = y
        self.w = w
        self.it = 0
        super().__init__(cache_prefix=cache)

    def next(self, input_data: Callable) -> int:
        if self.it == len(self.X):
            return 0
        with pytest.raises(TypeError, match='Keyword argument'):
            input_data(self.X[self.it], self.y[self.it], None)
        input_data(data=self.X[self.it].copy(), label=self.y[self.it].copy(), weight=self.w[self.it].copy() if self.w else None)
        gc.collect()
        self.it += 1
        return 1

    def reset(self) -> None:
        self.it = 0

    def as_arrays(self) -> Tuple[Union[np.ndarray, sparse.csr_matrix], ArrayLike, Optional[ArrayLike]]:
        if isinstance(self.X[0], sparse.csr_matrix):
            X = sparse.vstack(self.X, format='csr')
        else:
            X = np.concatenate(self.X, axis=0)
        y = np.concatenate(self.y, axis=0)
        if self.w:
            w = np.concatenate(self.w, axis=0)
        else:
            w = None
        return (X, y, w)