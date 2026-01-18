from __future__ import absolute_import
from six.moves import input
from decimal import Decimal
import re
from gslib.exception import CommandException
from gslib.lazy_wrapper import LazyWrapper
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
def DaysToSeconds(days):
    """Converts duration specified in days to equivalent seconds.

  Args:
    days: Retention duration in number of days.

  Returns:
    Returns the equivalent duration in seconds.
  """
    return days * SECONDS_IN_DAY