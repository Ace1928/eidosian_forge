from __future__ import absolute_import
import time
from socket import _GLOBAL_DEFAULT_TIMEOUT
from ..exceptions import TimeoutStateError
@classmethod
def _validate_timeout(cls, value, name):
    """Check that a timeout attribute is valid.

        :param value: The timeout value to validate
        :param name: The name of the timeout attribute to validate. This is
            used to specify in error messages.
        :return: The validated and casted version of the given value.
        :raises ValueError: If it is a numeric value less than or equal to
            zero, or the type is not an integer, float, or None.
        """
    if value is _Default:
        return cls.DEFAULT_TIMEOUT
    if value is None or value is cls.DEFAULT_TIMEOUT:
        return value
    if isinstance(value, bool):
        raise ValueError('Timeout cannot be a boolean value. It must be an int, float or None.')
    try:
        float(value)
    except (TypeError, ValueError):
        raise ValueError('Timeout value %s was %s, but it must be an int, float or None.' % (name, value))
    try:
        if value <= 0:
            raise ValueError('Attempted to set %s timeout to %s, but the timeout cannot be set to a value less than or equal to 0.' % (name, value))
    except TypeError:
        raise ValueError('Timeout value %s was %s, but it must be an int, float or None.' % (name, value))
    return value