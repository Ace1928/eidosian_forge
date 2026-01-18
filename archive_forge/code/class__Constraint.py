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
class _Constraint(ABC):
    """Base class for the constraint objects."""

    def __init__(self):
        self.hidden = False

    @abstractmethod
    def is_satisfied_by(self, val):
        """Whether or not a value satisfies the constraint.

        Parameters
        ----------
        val : object
            The value to check.

        Returns
        -------
        is_satisfied : bool
            Whether or not the constraint is satisfied by this value.
        """

    @abstractmethod
    def __str__(self):
        """A human readable representational string of the constraint."""