import itertools
from collections import Counter, defaultdict
from functools import reduce, wraps
@property
def _sgn_colors(self):
    if self._sgn_colors_ is None:
        self._sgn_colors_ = partition_to_color(self._sgn_partitions)
    return self._sgn_colors_