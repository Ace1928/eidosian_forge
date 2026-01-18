from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import threading
class WorkloadEstimatorMessage(ThreadMessage):
    """Message class for estimating total workload of operation.

  Attributes:
    item_count (int): Number of items to add to workload estimation.
    size (int|None): Number of bytes to add to workload estimation.
  """

    def __init__(self, item_count, size=None):
        """Initializes WorkloadEstimatorMessage. Args in attributes docstring."""
        self.item_count = item_count
        self.size = size

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.__dict__ == other.__dict__

    def __repr__(self):
        """Returns a string with a valid constructor for this message."""
        return '{class_name}(item_count={item_count}, size={size})'.format(class_name=self.__class__.__name__, item_count=self.item_count, size=self.size)