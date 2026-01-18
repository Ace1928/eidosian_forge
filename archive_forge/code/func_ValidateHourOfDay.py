from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import ipaddress
import re
from typing import Any
from googlecloudsdk.calliope import exceptions
def ValidateHourOfDay(hour):
    """Validates that the hour falls between 0 and 23, inclusive."""
    if hour < 0 or hour > 23:
        raise exceptions.BadArgumentException('--maintenance-window-hour-of-day', 'Hour of day ({0}) is not in [0, 23].'.format(hour))
    return hour