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
def _mark_if_deprecated(self, option):
    """Add a deprecated mark to an option if needed."""
    option_str = f'{option!r}'
    if option in self.deprecated:
        option_str = f'{option_str} (deprecated)'
    return option_str