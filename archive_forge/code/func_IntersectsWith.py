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
def IntersectsWith(self, other):
    """Determines if this platform intersects with the other platform.

    Platforms intersect if they can both potentially be installed on the same
    system.

    Args:
      other: ComponentPlatform, The other component platform to compare against.

    Returns:
      bool, True if there is any intersection, False otherwise.
    """
    return self.__CollectionsIntersect(self.operating_systems, other.operating_systems) and self.__CollectionsIntersect(self.architectures, other.architectures)