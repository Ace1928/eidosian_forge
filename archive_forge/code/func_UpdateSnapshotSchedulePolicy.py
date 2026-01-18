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
def UpdateSnapshotSchedulePolicy(self, policy_resource, description, labels, schedules):
    """Sends request to update an existing SnapshotSchedulePolicy."""
    updated_fields = []
    if description:
        updated_fields.append('description')
    if labels is not None:
        updated_fields.append('labels')
    schedule_msgs = self._ParseSnapshotSchedules(schedules)
    if schedule_msgs:
        updated_fields.append('schedules')
    update_mask = ','.join(updated_fields)
    policy_msg = self.messages.SnapshotSchedulePolicy(description=description, schedules=schedule_msgs, labels=labels)
    request = self.messages.BaremetalsolutionProjectsLocationsSnapshotSchedulePoliciesPatchRequest(name=policy_resource.RelativeName(), snapshotSchedulePolicy=policy_msg, updateMask=update_mask)
    return self.snapshot_schedule_policies_service.Patch(request)