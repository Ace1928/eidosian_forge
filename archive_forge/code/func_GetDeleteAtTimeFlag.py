from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
def GetDeleteAtTimeFlag():
    """Gets the --delete-at-time flag."""
    help_text = '  Automatically deletes the reservation at a specific time from its creation.\n  The specified time must be an RFC3339 timestamp, which must be\n  formatted as "YYYY-MM-DDTHH:MM:SSZ" where YYYY = year, MM = month, DD = day,\n  HH = hours, MM = minutes, SS = seconds, and Z = time zone in\n  Coordinated Universal Time (UTC). For example, specify 2021-11-20T07:00:00Z.\n  '
    return base.Argument('--delete-at-time', type=arg_parsers.Datetime.Parse, help=help_text)