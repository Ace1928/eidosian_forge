from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import heapq
import math
import os
import tensorflow as tf
from tensorflow.python.platform import gfile
def _mod_export_version(n):
    """Creates a filter that keeps every export that is a multiple of n.

  Args:
    n: step size.

  Returns:
    A filter function that keeps paths where export_version % n == 0.
  """

    def keep(paths):
        keepers = []
        for p in paths:
            if p.export_version % n == 0:
                keepers.append(p)
        return sorted(keepers)
    return keep