from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import exceptions as apilib_exceptions
from googlecloudsdk.command_lib.bms import util
import six
def _ParseSnapshotSchedules(self, schedules):
    """Parses schedule ArgDict dicts into a list of Schedule messages."""
    schedule_msgs = []
    if schedules:
        for schedule_arg in schedules:
            schedule_msgs.append(self.messages.Schedule(crontabSpec=schedule_arg['crontab_spec'], retentionCount=schedule_arg['retention_count'], prefix=schedule_arg['prefix']))
    return schedule_msgs