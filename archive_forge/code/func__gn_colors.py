import itertools
from collections import Counter, defaultdict
from functools import reduce, wraps
@property
def _gn_colors(self):
    if self._gn_colors_ is None:
        self._gn_colors_ = partition_to_color(self._gn_partitions)
    return self._gn_colors_