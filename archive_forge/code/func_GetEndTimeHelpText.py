from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
from googlecloudsdk.command_lib.compute.reservations import flags as reservation_flags
def GetEndTimeHelpText():
    """Gets the --end-time help text."""
    help_text = '  End time of the Future Reservation. The end time must be an RFC3339 valid\n  string formatted by date, time, and timezone or "YYYY-MM-DDTHH:MM:SSZ"; where\n  YYYY = year, MM = month, DD = day, HH = hours, MM = minutes, SS = seconds, and\n  Z = timezone (i.e. 2021-11-20T07:00:00Z).\n  '
    return help_text