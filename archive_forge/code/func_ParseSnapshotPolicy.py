from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.netapp import constants
from googlecloudsdk.api_lib.netapp import util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def ParseSnapshotPolicy(self, volume, snapshot_policy):
    """Parses Snapshot Policy from a list of snapshot schedules into a given Volume.

    Args:
      volume: The Cloud NetApp Volume message object
      snapshot_policy: A list of snapshot policies (schedules) to parse

    Returns:
      Volume messages populated with snapshotPolicy field
    """
    if not snapshot_policy:
        return
    volume.snapshotPolicy = self.messages.SnapshotPolicy()
    volume.snapshotPolicy.enabled = True
    for name, snapshot_schedule in snapshot_policy.items():
        if name == 'hourly_snapshot':
            schedule = self.messages.HourlySchedule()
            schedule.snapshotsToKeep = snapshot_schedule.get('snapshots-to-keep')
            schedule.minute = snapshot_schedule.get('minute', 0)
            volume.snapshotPolicy.hourlySchedule = schedule
        elif name == 'daily_snapshot':
            schedule = self.messages.DailySchedule()
            schedule.snapshotsToKeep = snapshot_schedule.get('snapshots-to-keep')
            schedule.minute = snapshot_schedule.get('minute', 0)
            schedule.hour = snapshot_schedule.get('hour', 0)
            volume.snapshotPolicy.dailySchedule = schedule
        elif name == 'weekly_snapshot':
            schedule = self.messages.WeeklySchedule()
            schedule.snapshotsToKeep = snapshot_schedule.get('snapshots-to-keep')
            schedule.minute = snapshot_schedule.get('minute', 0)
            schedule.hour = snapshot_schedule.get('hour', 0)
            schedule.day = snapshot_schedule.get('day', 'Sunday')
            volume.snapshotPolicy.weeklySchedule = schedule
        elif name == 'monthly-snapshot':
            schedule = self.messages.MonthlySchedule()
            schedule.snapshotsToKeep = snapshot_schedule.get('snapshots-to-keep')
            schedule.minute = snapshot_schedule.get('minute', 0)
            schedule.hour = snapshot_schedule.get('hour', 0)
            schedule.day = snapshot_schedule.get('day', 1)
            volume.snapshotPolicy.monthlySchedule = schedule