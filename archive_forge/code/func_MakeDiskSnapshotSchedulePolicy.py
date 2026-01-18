from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import scope as compute_scopes
from googlecloudsdk.command_lib.compute.resource_policies import flags
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import times
def MakeDiskSnapshotSchedulePolicy(policy_ref, args, messages):
    """Creates a Disk Snapshot Schedule Resource Policy message from args."""
    hourly_cycle, daily_cycle, weekly_cycle = _ParseCycleFrequencyArgs(args, messages, supports_hourly=True, supports_weekly=True)
    snapshot_properties = None
    snapshot_labels = labels_util.ParseCreateArgs(args, messages.ResourcePolicySnapshotSchedulePolicySnapshotProperties.LabelsValue, labels_dest='snapshot_labels')
    storage_location = [args.storage_location] if args.storage_location else []
    if args.IsSpecified('guest_flush') or snapshot_labels or storage_location:
        snapshot_properties = messages.ResourcePolicySnapshotSchedulePolicySnapshotProperties(guestFlush=args.guest_flush, labels=snapshot_labels, storageLocations=storage_location)
    snapshot_policy = messages.ResourcePolicySnapshotSchedulePolicy(retentionPolicy=messages.ResourcePolicySnapshotSchedulePolicyRetentionPolicy(maxRetentionDays=args.max_retention_days, onSourceDiskDelete=flags.GetOnSourceDiskDeleteFlagMapper(messages).GetEnumForChoice(args.on_source_disk_delete)), schedule=messages.ResourcePolicySnapshotSchedulePolicySchedule(hourlySchedule=hourly_cycle, dailySchedule=daily_cycle, weeklySchedule=weekly_cycle), snapshotProperties=snapshot_properties)
    return messages.ResourcePolicy(name=policy_ref.Name(), description=args.description, region=policy_ref.region, snapshotSchedulePolicy=snapshot_policy)