from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import re
from dateutil import parser
from dateutil import tz
from dateutil.tz import _common as tz_common
import enum
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import iso_duration
from googlecloudsdk.core.util import times_data
import six
def _StrPtime(string, fmt):
    """Convert strptime exceptions to Datetime Errors."""
    try:
        return datetime.datetime.strptime(string, fmt)
    except (AttributeError, OverflowError, TypeError) as e:
        raise DateTimeValueError(six.text_type(e))
    except ValueError as e:
        raise DateTimeSyntaxError(six.text_type(e))