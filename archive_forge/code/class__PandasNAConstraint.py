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
class _PandasNAConstraint(_Constraint):
    """Constraint representing the indicator `pd.NA`."""

    def is_satisfied_by(self, val):
        try:
            import pandas as pd
            return isinstance(val, type(pd.NA)) and pd.isna(val)
        except ImportError:
            return False

    def __str__(self):
        return 'pandas.NA'