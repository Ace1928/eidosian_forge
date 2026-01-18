from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import sys
def _Defaults(self, projection):
    """Defaults() helper -- converts a projection to a default projection.

    Args:
      projection: A node in the original projection _Tree.
    """
    projection.attribute.flag = self.DEFAULT
    for node in projection.tree.values():
        self._Defaults(node)