from __future__ import absolute_import
from six.moves import input
from decimal import Decimal
import re
from gslib.exception import CommandException
from gslib.lazy_wrapper import LazyWrapper
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
def _RetentionPeriodToString(retention_period):
    """Converts Retention Period to Human readable format.

  Args:
    retention_period: Retention duration in seconds (integer value).

  Returns:
    Returns a string representing retention duration in human readable format.
  """
    period = Decimal(retention_period)
    duration_str = None
    if period // SECONDS_IN_YEAR == period / SECONDS_IN_YEAR:
        duration_str = '{} Year(s)'.format(period // SECONDS_IN_YEAR)
    elif period // SECONDS_IN_MONTH == period / SECONDS_IN_MONTH:
        duration_str = '{} Month(s)'.format(period // SECONDS_IN_MONTH)
    elif period // SECONDS_IN_DAY == period / SECONDS_IN_DAY:
        duration_str = '{} Day(s)'.format(period // SECONDS_IN_DAY)
    elif period > SECONDS_IN_DAY:
        duration_str = '{} Seconds (~{} Day(s))'.format(retention_period, period // SECONDS_IN_DAY)
    else:
        duration_str = '{} Second(s)'.format(retention_period)
    return '    Duration: {}'.format(duration_str)