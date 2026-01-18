from __future__ import absolute_import
from six.moves import input
from decimal import Decimal
import re
from gslib.exception import CommandException
from gslib.lazy_wrapper import LazyWrapper
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
def YearsToSeconds(years):
    """Converts duration specified in years to equivalent seconds.

    GCS bucket lock API uses following duration equivalencies to convert
    durations specified in terms of months or years to seconds:
      - A month is considered to be 31 days or 2,678,400 seconds.
      - A year is considered to be 365.25 days or 31,557,600 seconds.

  Args:
    years: Retention duration in number of years.

  Returns:
    Returns the rough equivalent duration in seconds.
  """
    return years * SECONDS_IN_YEAR