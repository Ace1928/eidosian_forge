from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import sys
def Columns(self):
    """Returns the projection columns.

    Returns:
      The columns in the projection, None if the entire resource is projected.
    """
    return self._columns