from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import sys
def Order(self):
    """Returns the projection sort key order suitable for use by sorted().

    Example:
      projection = resource_projector.Compile('...')
      order = projection.Order()
      if order:
        rows = sorted(rows, key=itemgetter(*order))

    Returns:
      The list of (sort-key-index, reverse), [] if projection is None
      or if all sort order indices in the projection are None (unordered).
    """
    ordering = []
    for i, col in enumerate(self._columns):
        if col.attribute.order or col.attribute.reverse:
            ordering.append((col.attribute.order or sys.maxsize, i, col.attribute.reverse))
    return [(i, reverse) for _, i, reverse in sorted(ordering)]