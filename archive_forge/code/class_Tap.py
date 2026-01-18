from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
class Tap(object):
    """A Tapper Tap object."""

    def Tap(self, item):
        """Called on each item as it is fetched.

    Args:
      item: The current item to be tapped.

    Returns:
      True: The item is retained in the iterable.
      False: The item is deleted from the iterable.
      None: The item is deleted from the iterable and the iteration stops.
      Injector(): Injector.value is injected into the iterable. If
        Injector.is_replacement then the item is deleted from the iterable,
        otherwise the item appears in the iterable after the injected value.
    """
        _ = item
        return True

    def Done(self):
        """Called after the last item."""
        pass