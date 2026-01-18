from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import encoding
from googlecloudsdk.calliope import exceptions as calliope_exc
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import times
import six
def ModifySnooze(base_snooze, messages, display_name=None, criteria_policies=None, start_time=None, end_time=None, field_masks=None):
    """Override and/or add fields from other flags to an Snooze."""
    if field_masks is None:
        field_masks = []
    start_time_target = None
    start_time_from_base = False
    if start_time is not None:
        field_masks.append('interval.start_time')
        start_time_target = start_time
    else:
        try:
            start_time_target = times.ParseDateTime(base_snooze.interval.startTime)
            start_time_from_base = True
        except AttributeError:
            pass
    end_time_target = None
    end_time_from_base = False
    if end_time is not None:
        field_masks.append('interval.end_time')
        end_time_target = end_time
    else:
        try:
            end_time_target = times.ParseDateTime(base_snooze.interval.endTime)
            end_time_from_base = True
        except AttributeError:
            pass
    try:
        if start_time_target is not None and (not start_time_from_base):
            base_snooze.interval.startTime = times.FormatDateTime(start_time_target)
        if end_time_target is not None and (not end_time_from_base):
            base_snooze.interval.endTime = times.FormatDateTime(end_time_target)
    except AttributeError:
        interval = messages.TimeInterval()
        interval.startTime = times.FormatDateTime(start_time_target)
        interval.endTime = times.FormatDateTime(end_time_target)
        base_snooze.interval = interval
    if display_name is not None:
        field_masks.append('display_name')
        base_snooze.displayName = display_name
    if criteria_policies is not None:
        field_masks.append('criteria_policies')
        criteria = messages.Criteria()
        criteria.policies = criteria_policies
        base_snooze.criteria = criteria