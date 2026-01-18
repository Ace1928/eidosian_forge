from __future__ import absolute_import
from six.moves import input
from decimal import Decimal
import re
from gslib.exception import CommandException
from gslib.lazy_wrapper import LazyWrapper
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
def RetentionInDaysMatch(days):
    """Test whether the string matches retention in days pattern.

  Args:
    days: string to match for retention specified in days format.

  Returns:
    Returns a match object if the string matches the retention in days
    pattern. The match object will contain a 'number' group for the duration
    in number of days. Otherwise, None is returned.
  """
    return _RETENTION_IN_DAYS().match(days)