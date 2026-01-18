from collections import Counter
from contextlib import suppress
from typing import NamedTuple
import numpy as np
from . import is_scalar_nan
class _NaNCounter(Counter):
    """Counter with support for nan values."""

    def __init__(self, items):
        super().__init__(self._generate_items(items))

    def _generate_items(self, items):
        """Generate items without nans. Stores the nan counts separately."""
        for item in items:
            if not is_scalar_nan(item):
                yield item
                continue
            if not hasattr(self, 'nan_count'):
                self.nan_count = 0
            self.nan_count += 1

    def __missing__(self, key):
        if hasattr(self, 'nan_count') and is_scalar_nan(key):
            return self.nan_count
        raise KeyError(key)