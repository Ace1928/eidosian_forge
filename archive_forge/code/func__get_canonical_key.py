from __future__ import annotations
from contextlib import contextmanager
from typing import (
from pandas.plotting._core import _get_plot_backend
def _get_canonical_key(self, key):
    return self._ALIASES.get(key, key)