from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import numpy as np
import tensorflow as tf
def _mean_runtime_secs(self):
    return np.mean(self._buffer_wheel, axis=0)[0] if self._buffer_wheel else 0