from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
import time
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core.util import platforms
from googlecloudsdk.core.util import semver
import six
def __CollectionsIntersect(self, collection1, collection2):
    """Determines if the two collections intersect.

    The collections intersect if either or both are None or empty, or if they
    contain an intersection of elements.

    Args:
      collection1: [] or None, The first collection.
      collection2: [] or None, The second collection.

    Returns:
      bool, True if there is an intersection, False otherwise.
    """
    if not collection1 or not collection2:
        return True
    return set(collection1) & set(collection2)