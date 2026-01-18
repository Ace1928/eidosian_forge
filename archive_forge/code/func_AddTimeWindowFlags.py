from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
from googlecloudsdk.command_lib.compute.reservations import flags as reservation_flags
def AddTimeWindowFlags(parser, time_window_requird=False):
    """Adds all flags needed for the modifying the time window properties."""
    time_window_group = parser.add_group(help='Manage the time specific properties for requesting future capacity', required=time_window_requird)
    time_window_group.add_argument('--start-time', required=time_window_requird, help=GetStartTimeHelpText())
    end_time_window_group = time_window_group.add_mutually_exclusive_group(required=time_window_requird)
    end_time_window_group.add_argument('--end-time', help=GetEndTimeHelpText())
    end_time_window_group.add_argument('--duration', type=int, help=GetDurationHelpText())