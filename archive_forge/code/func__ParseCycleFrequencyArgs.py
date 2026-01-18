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
def _ParseCycleFrequencyArgs(args, messages, supports_hourly=False, supports_weekly=False):
    """Parses args and returns a tuple of DailyCycle and WeeklyCycle messages."""
    _ValidateCycleFrequencyArgs(args)
    hourly_cycle, daily_cycle, weekly_cycle = (None, None, None)
    if args.daily_cycle:
        daily_cycle = messages.ResourcePolicyDailyCycle(daysInCycle=1, startTime=_FormatStartTime(args.start_time))
    if supports_weekly:
        if args.weekly_cycle:
            day_enum = messages.ResourcePolicyWeeklyCycleDayOfWeek.DayValueValuesEnum
            weekday = times.Weekday.Get(args.weekly_cycle.upper())
            day, start_time = _ParseWeeklyDayAndTime(args.start_time, weekday)
            weekly_cycle = messages.ResourcePolicyWeeklyCycle(dayOfWeeks=[messages.ResourcePolicyWeeklyCycleDayOfWeek(day=day_enum(day), startTime=start_time)])
        if args.IsSpecified('weekly_cycle_from_file'):
            if args.weekly_cycle_from_file:
                weekly_cycle = _ParseWeeklyCycleFromFile(args, messages)
            else:
                raise exceptions.InvalidArgumentException(args.GetFlag('weekly_cycle_from_file'), 'File cannot be empty.')
    if supports_hourly and args.hourly_cycle:
        hourly_cycle = messages.ResourcePolicyHourlyCycle(hoursInCycle=args.hourly_cycle, startTime=_FormatStartTime(args.start_time))
    return (hourly_cycle, daily_cycle, weekly_cycle)