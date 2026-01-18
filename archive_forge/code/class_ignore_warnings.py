from __future__ import annotations
import inspect
import itertools
import warnings
from collections import defaultdict
from collections.abc import Iterable, Sequence
from contextlib import suppress
from copy import deepcopy
from dataclasses import field
from typing import TYPE_CHECKING, cast, overload
from warnings import warn
import numpy as np
import pandas as pd
from pandas.core.groupby import DataFrameGroupBy
from ..exceptions import PlotnineError, PlotnineWarning
from ..mapping import aes
class ignore_warnings:
    """
    Ignore Warnings Context Manager

    Wrap around warnings.catch_warnings to make ignoring
    warnings easier.

    Parameters
    ----------
    *categories : tuple
        Warning categories to ignore e.g UserWarning,
        FutureWarning, RuntimeWarning, ...
    """
    _cm: warnings.catch_warnings

    def __init__(self, *categories):
        self.categories = categories
        self._cm = warnings.catch_warnings()

    def __enter__(self):
        self._cm.__enter__()
        for c in self.categories:
            warnings.filterwarnings('ignore', category=c)

    def __exit__(self, type, value, traceback):
        return self._cm.__exit__(type, value, traceback)