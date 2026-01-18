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
def CreateSnapshotSchedulePolicy(self, policy_resource, labels, description, schedules):
    """Sends request to create a new Snapshot Schedule Policy."""
    policy_id = policy_resource.Name()
    parent = policy_resource.Parent().RelativeName()
    schedule_msgs = self._ParseSnapshotSchedules(schedules)
    policy_msg = self.messages.SnapshotSchedulePolicy(description=description, schedules=schedule_msgs, labels=labels)
    request = self.messages.BaremetalsolutionProjectsLocationsSnapshotSchedulePoliciesCreateRequest(parent=parent, snapshotSchedulePolicyId=policy_id, snapshotSchedulePolicy=policy_msg)
    return self.snapshot_schedule_policies_service.Create(request)