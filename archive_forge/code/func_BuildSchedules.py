from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import random
import re
import string
import sys
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.api_lib.compute import request_helper
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.managed_instance_groups import auto_healing_utils
from googlecloudsdk.command_lib.compute.managed_instance_groups import update_instances_utils
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
from six.moves import range  # pylint: disable=redefined-builtin
def BuildSchedules(args, messages):
    """Builds AutoscalingPolicyScalingSchedules.

  Args:
    args: command line arguments.
    messages: module containing message classes.

  Returns:
    Dict containing an AutoscalingPolicyScalingSchedule message object.
  Raises:
    InvalidArgumentError:  if more than one of --scaling-schedule,
    --update-schedule, --remove-schedule,
    --enable-schedule, --disable-schedule is specified.
  """
    mutex_group = {'set_schedule', 'update_schedule', 'remove_schedule', 'enable_schedule', 'disable_schedule'}
    count = 0
    for possible_argument in mutex_group:
        if getattr(args, possible_argument, None) is not None:
            count += 1
    if count == 0:
        return None
    if count > 1:
        raise InvalidArgumentError('--set-schedule, --update-schedule, --remove-schedule, --enable-schedule, --disable-schedule are mutually exclusive, only one can be specified.')
    scaling_schedule_wrapper = messages.AutoscalingPolicy.ScalingSchedulesValue.AdditionalProperty
    field_mapping = {'schedule_cron': 'schedule', 'schedule_duration_sec': 'durationSec', 'schedule_min_required_replicas': 'minRequiredReplicas', 'schedule_time_zone': 'timeZone', 'schedule_description': 'description'}
    if getattr(args, 'enable_schedule', None) is not None:
        return messages.AutoscalingPolicy.ScalingSchedulesValue(additionalProperties=[scaling_schedule_wrapper(key=args.enable_schedule, value=messages.AutoscalingPolicyScalingSchedule(disabled=False))])
    if getattr(args, 'disable_schedule', None) is not None:
        return messages.AutoscalingPolicy.ScalingSchedulesValue(additionalProperties=[scaling_schedule_wrapper(key=args.disable_schedule, value=messages.AutoscalingPolicyScalingSchedule(disabled=True))])
    if getattr(args, 'remove_schedule', None) is not None:
        encoding.RegisterCustomMessageCodec(encoder=_RemoveScheduleEncoder, decoder=_RemoveScheduleDecoder)(messages.AutoscalingPolicy.ScalingSchedulesValue)
        return messages.AutoscalingPolicy.ScalingSchedulesValue(additionalProperties=[scaling_schedule_wrapper(key=args.remove_schedule, value=None)])
    if getattr(args, 'set_schedule', None) is not None:
        policy_name = args.set_schedule
        required = {'schedule_cron', 'schedule_duration_sec', 'schedule_min_required_replicas'}
        scaling_schedule = {field: None for field in field_mapping.values()}
    else:
        policy_name = args.update_schedule
        required = set()
        scaling_schedule = {}
    for arg_attr, field in six.iteritems(field_mapping):
        arg = getattr(args, arg_attr, None)
        if arg is not None:
            scaling_schedule[field] = arg
        elif arg_attr in required:
            raise InvalidArgumentError('--set-schedule argument requires --schedule-duration-sec, --schedule-cron, and --schedule-min-required-replicas to be specified.')
    return messages.AutoscalingPolicy.ScalingSchedulesValue(additionalProperties=[scaling_schedule_wrapper(key=policy_name, value=messages.AutoscalingPolicyScalingSchedule(**scaling_schedule))])