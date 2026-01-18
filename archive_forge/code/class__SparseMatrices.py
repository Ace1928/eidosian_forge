import functools
import math
import operator
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable
from inspect import signature
from numbers import Integral, Real
import numpy as np
from scipy.sparse import csr_matrix, issparse
from .._config import config_context, get_config
from .validation import _is_arraylike_not_scalar
class _SparseMatrices(_Constraint):
    """Constraint representing sparse matrices."""

    def is_satisfied_by(self, val):
        return issparse(val)

    def __str__(self):
        return 'a sparse matrix'