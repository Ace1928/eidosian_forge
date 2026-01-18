from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import log
from googlecloudsdk.core.resource import resource_filter
from googlecloudsdk.core.resource import resource_printer_base
from googlecloudsdk.core.resource import resource_projector
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import peek_iterable
class Limiter(peek_iterable.Tap):
    """A Tapper class that filters out resources after a limit is reached.

  Attributes:
    _limit: The resource count limit.
    _count: The resource count.
  """

    def __init__(self, limit):
        self._limit = limit
        self._count = 0

    def Tap(self, resource):
        """Returns True if the limit has not been reached yet, None otherwise.

    Args:
      resource: The resource to limit.

    Returns:
      True if the limit has not been reached yet, None otherwise to stop
      iterations.
    """
        if resource_printer_base.IsResourceMarker(resource):
            return True
        self._count += 1
        return self._count <= self._limit or None