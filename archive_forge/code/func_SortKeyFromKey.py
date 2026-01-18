from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import collections
from collections.abc import Container, Mapping
from googlecloudsdk.core import exceptions
def SortKeyFromKey(key):
    """Sorted key function  to order TrafficTarget keys.

  TrafficTargets keys are one of:
  o revisionName
  o LATEST_REVISION_KEY

  Note LATEST_REVISION_KEY is not a str so its ordering with respect
  to revisionName keys is hard to predict.

  Args:
    key: Key for a TrafficTargets dictionary.

  Returns:
    A value that sorts by revisionName with LATEST_REVISION_KEY
    last.
  """
    if key == LATEST_REVISION_KEY:
        result = (2, key)
    else:
        result = (1, key)
    return result