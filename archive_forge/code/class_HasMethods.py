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
class HasMethods(_Constraint):
    """Constraint representing objects that expose specific methods.

    It is useful for parameters following a protocol and where we don't want to impose
    an affiliation to a specific module or class.

    Parameters
    ----------
    methods : str or list of str
        The method(s) that the object is expected to expose.
    """

    @validate_params({'methods': [str, list]}, prefer_skip_nested_validation=True)
    def __init__(self, methods):
        super().__init__()
        if isinstance(methods, str):
            methods = [methods]
        self.methods = methods

    def is_satisfied_by(self, val):
        return all((callable(getattr(val, method, None)) for method in self.methods))

    def __str__(self):
        if len(self.methods) == 1:
            methods = f'{self.methods[0]!r}'
        else:
            methods = f'{', '.join([repr(m) for m in self.methods[:-1]])} and {self.methods[-1]!r}'
        return f'an object implementing {methods}'