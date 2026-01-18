from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import heapq
import math
import os
import tensorflow as tf
from tensorflow.python.platform import gfile
def _one_of_every_n_export_versions(n):
    """Creates a filter that keeps one of every n export versions.

  Args:
    n: interval size.

  Returns:
    A filter function that keeps exactly one path from each interval
    [0, n], (n, 2n], (2n, 3n], etc...  If more than one path exists in an
    interval the largest is kept.
  """

    def keep(paths):
        """A filter function that keeps exactly one out of every n paths."""
        keeper_map = {}
        for p in paths:
            if p.export_version is None:
                continue
            interval = math.floor((p.export_version - 1) / n) if p.export_version else 0
            existing = keeper_map.get(interval, None)
            if not existing or existing.export_version < p.export_version:
                keeper_map[interval] = p
        return sorted(keeper_map.values())
    return keep