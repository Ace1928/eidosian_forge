from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
from googlecloudsdk.command_lib.compute.reservations import flags as reservation_flags
def GetAutoCreatedReservationsDurationFlag(required=False):
    """Gets the --auto-created-reservations-duration flag."""
    help_text = '  Automatically deletes an auto-created reservations after a specified\n  number of days, hours, minutes, or seconds. For example, specify 30m\n  for 30 minutes, or 1d2h3m4s for 1 day, 2 hours, 3 minutes, and 4\n  seconds. For more information, see $ gcloud topic datetimes.\n  '
    return base.Argument('--auto-created-reservations-duration', required=required, type=arg_parsers.Duration(), help=help_text)