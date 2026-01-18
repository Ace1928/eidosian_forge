from __future__ import annotations
import contextlib
import functools
import inspect
import io
import itertools
import math
import os
import re
import sys
import warnings
from collections.abc import (
from enum import Enum
from pathlib import Path
from typing import (
import numpy as np
import pandas as pd
from xarray.namedarray.utils import (  # noqa: F401
class NdimSizeLenMixin:
    """Mixin class that extends a class that defines a ``shape`` property to
    one that also defines ``ndim``, ``size`` and ``__len__``.
    """
    __slots__ = ()

    @property
    def ndim(self: Any) -> int:
        """
        Number of array dimensions.

        See Also
        --------
        numpy.ndarray.ndim
        """
        return len(self.shape)

    @property
    def size(self: Any) -> int:
        """
        Number of elements in the array.

        Equal to ``np.prod(a.shape)``, i.e., the product of the array’s dimensions.

        See Also
        --------
        numpy.ndarray.size
        """
        return math.prod(self.shape)

    def __len__(self: Any) -> int:
        try:
            return self.shape[0]
        except IndexError:
            raise TypeError('len() of unsized object')