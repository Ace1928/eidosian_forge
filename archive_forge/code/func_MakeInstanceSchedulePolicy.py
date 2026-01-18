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
def MakeInstanceSchedulePolicy(policy_ref, args, messages):
    """Creates an Instance Schedule Policy message from args."""
    vm_start_schedule = None
    if args.vm_start_schedule:
        vm_start_schedule = messages.ResourcePolicyInstanceSchedulePolicySchedule(schedule=args.vm_start_schedule)
    vm_stop_schedule = None
    if args.vm_stop_schedule:
        vm_stop_schedule = messages.ResourcePolicyInstanceSchedulePolicySchedule(schedule=args.vm_stop_schedule)
    instance_schedule_policy = messages.ResourcePolicyInstanceSchedulePolicy(timeZone=args.timezone, vmStartSchedule=vm_start_schedule, vmStopSchedule=vm_stop_schedule)
    if args.initiation_date:
        instance_schedule_policy.startTime = times.FormatDateTime(args.initiation_date)
    if args.end_date:
        instance_schedule_policy.expirationTime = times.FormatDateTime(args.end_date)
    return messages.ResourcePolicy(name=policy_ref.Name(), description=args.description, region=policy_ref.region, instanceSchedulePolicy=instance_schedule_policy)