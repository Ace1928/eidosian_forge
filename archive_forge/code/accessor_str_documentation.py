from __future__ import annotations
import codecs
import re
import textwrap
from collections.abc import Hashable, Mapping
from functools import reduce
from operator import or_ as set_union
from re import Pattern
from typing import TYPE_CHECKING, Any, Callable, Generic
from unicodedata import normalize
import numpy as np
from xarray.core import duck_array_ops
from xarray.core.computation import apply_ufunc
from xarray.core.types import T_DataArray

        Encode character string in the array using indicated encoding.

        Parameters
        ----------
        encoding : str
            The encoding to use.
            Please see the Python documentation `codecs standard encoders <https://docs.python.org/3/library/codecs.html#standard-encodings>`_
            section for a list of encodings handlers.
        errors : str, default: "strict"
            The handler for encoding errors.
            Please see the Python documentation `codecs error handlers <https://docs.python.org/3/library/codecs.html#error-handlers>`_
            for a list of error handlers.

        Returns
        -------
        encoded : same type as values
        