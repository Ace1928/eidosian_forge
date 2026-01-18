from __future__ import absolute_import
import re
def IdleTimeoutToDuration(value):
    """Convert valid idle timeout argument to a Duration value of seconds.

  Args:
    value: A string in the form Xm or Xs

  Returns:
    Duration value of the given argument.

  Raises:
    ValueError: if the given value isn't parseable.
  """
    if not re.compile(appinfo._IDLE_TIMEOUT_REGEX).match(value):
        raise ValueError('Unrecognized idle timeout: %s' % value)
    if value.endswith('m'):
        return '%ss' % (int(value[:-1]) * _SECONDS_PER_MINUTE)
    else:
        return value