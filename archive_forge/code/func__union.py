from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import heapq
import math
import os
import tensorflow as tf
from tensorflow.python.platform import gfile
def _union(lf, rf):
    """Creates a filter that keeps the union of two filters.

  Args:
    lf: first filter
    rf: second filter

  Returns:
    A filter function that keeps the n largest paths.
  """

    def keep(paths):
        l = set(lf(paths))
        r = set(rf(paths))
        return sorted(list(l | r))
    return keep