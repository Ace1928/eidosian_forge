from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def GetRunScheduleArg(for_update):
    help_str = 'Schedule for the crawler run.'
    if not for_update:
        help_str += ' This argument should be provided if and only if `--run-option=SCHEDULED` was specified.'
    else:
        help_str += ' This argument can be provided only if the crawler run option will be scheduled after updating.'
    return base.ChoiceArgument('--run-schedule', choices=['daily', 'weekly'], help_str=help_str)