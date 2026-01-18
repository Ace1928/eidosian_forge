from __future__ import annotations
from ._array_object import Array
from ._dtypes import _all_dtypes, _result_type
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Tuple, Union
import cupy as np

    Array API compatible wrapper for :py:func:`np.result_type <numpy.result_type>`.

    See its docstring for more information.
    