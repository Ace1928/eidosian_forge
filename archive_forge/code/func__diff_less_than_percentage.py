from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import numpy as np
import tensorflow as tf
def _diff_less_than_percentage(self, actual, target, percentage):
    """Checks if `actual` value is within a `percentage` to `target` value.

    Args:
      actual: Actual value.
      target: Target value.
      percentage: Max percentage threshold.

    Returns:
      True if the ABS(`actual` - `target`) is less than or equal to `percentage`
        , otherwise False.

    Raise:
      ValueError: If `total_secs` value is not positive.
    """
    if actual == 0:
        raise ValueError('Invalid `actual` value. Value must not be zero.')
    if target == 0:
        raise ValueError('Invalid `target` value. Value must not be zero.')
    return float(abs(target - actual)) / target <= percentage * 0.01