from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import numpy as np
import tensorflow as tf
def _is_step_time_stable(self):
    """Checks if the step time has stabilized.

    We define stability a function of small stdev and after running for some
    time.

    Returns:
      True if stability is reached, False otherwise.
    """
    std = self._std_step_time_secs()
    return std < 0.03 and self._sample_count > self._capacity