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
def _ParseWeeklyCycleFromFile(args, messages):
    """Parses WeeklyCycle message from file contents specified in args."""
    weekly_cycle_dict = yaml.load(args.weekly_cycle_from_file)
    day_enum = messages.ResourcePolicyWeeklyCycleDayOfWeek.DayValueValuesEnum
    days_of_week = []
    for day_and_time in weekly_cycle_dict:
        if 'day' not in day_and_time or 'startTime' not in day_and_time:
            raise exceptions.InvalidArgumentException(args.GetFlag('weekly_cycle_from_file'), 'Each JSON/YAML object in the list must have the following keys: [day, startTime].')
        day = day_and_time['day'].upper()
        try:
            weekday = times.Weekday.Get(day)
        except KeyError:
            raise exceptions.InvalidArgumentException(args.GetFlag('weekly_cycle_from_file'), 'Invalid value for `day`: [{}].'.format(day))
        start_time = arg_parsers.Datetime.ParseUtcTime(day_and_time['startTime'])
        day, start_time = _ParseWeeklyDayAndTime(start_time, weekday)
        days_of_week.append(messages.ResourcePolicyWeeklyCycleDayOfWeek(day=day_enum(day), startTime=start_time))
    return messages.ResourcePolicyWeeklyCycle(dayOfWeeks=days_of_week)