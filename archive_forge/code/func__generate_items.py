from collections import Counter
from contextlib import suppress
from typing import NamedTuple
import numpy as np
from . import is_scalar_nan
def _generate_items(self, items):
    """Generate items without nans. Stores the nan counts separately."""
    for item in items:
        if not is_scalar_nan(item):
            yield item
            continue
        if not hasattr(self, 'nan_count'):
            self.nan_count = 0
        self.nan_count += 1