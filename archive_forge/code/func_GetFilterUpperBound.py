from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import time
import enum
from googlecloudsdk.api_lib.logging import common as logging_common
from googlecloudsdk.core import log
from googlecloudsdk.core.util import times
def GetFilterUpperBound(self, now):
    """The log message filter which keeps out messages which are too new.

    Args:
        now: The current time, as a datetime object.

    Returns:
        The upper bound filter text that we should use.
    """
    tzinfo = times.ParseDateTime(self.timestamp).tzinfo
    now = now.replace(tzinfo=tzinfo)
    upper_bound = now - datetime.timedelta(seconds=5)
    return 'timestamp<"{0}"'.format(times.FormatDateTime(upper_bound, '%Y-%m-%dT%H:%M:%S.%6f%Ez'))