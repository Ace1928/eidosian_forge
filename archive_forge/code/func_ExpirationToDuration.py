from __future__ import absolute_import
import re
def ExpirationToDuration(value):
    """Convert valid expiration argument to a Duration value of seconds.

  Args:
    value: String that matches _DELTA_REGEX.

  Returns:
    Time delta expressed as a Duration.

  Raises:
    ValueError: if the given value isn't parseable.
  """
    if not re.compile(appinfo._EXPIRATION_REGEX).match(value):
        raise ValueError('Unrecognized expiration: %s' % value)
    delta = appinfo.ParseExpiration(value)
    return '%ss' % delta