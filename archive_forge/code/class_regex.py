from calendar import timegm
from datetime import datetime, time, timedelta
from email.utils import parsedate_tz, mktime_tz
import re
import aniso8601
import pytz
class regex(object):
    """Validate a string based on a regular expression.

    Example::

        parser = reqparse.RequestParser()
        parser.add_argument('example', type=inputs.regex('^[0-9]+$'))

    Input to the ``example`` argument will be rejected if it contains anything
    but numbers.

    :param pattern: The regular expression the input must match
    :type pattern: str
    :param flags: Flags to change expression behavior
    :type flags: int
    """

    def __init__(self, pattern, flags=0):
        self.pattern = pattern
        self.re = re.compile(pattern, flags)

    def __call__(self, value):
        if not self.re.search(value):
            message = 'Value does not match pattern: "{0}"'.format(self.pattern)
            raise ValueError(message)
        return value

    def __deepcopy__(self, memo):
        return regex(self.pattern)