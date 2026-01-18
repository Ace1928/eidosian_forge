from __future__ import absolute_import
from six.moves import input
from decimal import Decimal
import re
from gslib.exception import CommandException
from gslib.lazy_wrapper import LazyWrapper
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
def RetentionInYearsMatch(years):
    """Test whether the string matches retention in years pattern.

  Args:
    years: string to match for retention specified in years format.

  Returns:
    Returns a match object if the string matches the retention in years
    pattern. The match object will contain a 'number' group for the duration
    in number of years. Otherwise, None is returned.
  """
    return _RETENTION_IN_YEARS().match(years)