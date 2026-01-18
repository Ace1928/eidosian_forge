import datetime
import uuid
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core.util import times
def ConvertUtcTime(effective_time):
    """Converts the date to UTC time.

  Args:
    effective_time: Date to be converted to UTC time.

  Returns:
    UTC time.

  Raises:
    ArgumentTypeError: If the date is not in the future.
  """
    if effective_time is None:
        return None
    if effective_time < times.Now().date():
        raise exceptions.InvalidArgumentException('Date must be in the future: {0}'.format(effective_time), 'effective_time')
    year = effective_time.year
    month = effective_time.month
    day = effective_time.day
    effective_time = datetime.datetime(year, month, day, 0, 0, 0, 0, datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    return effective_time