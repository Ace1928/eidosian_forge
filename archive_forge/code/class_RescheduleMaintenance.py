from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.sql import api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.sql import flags
from googlecloudsdk.command_lib.sql import reschedule_maintenance_util
@base.ReleaseTracks(base.ReleaseTrack.GA, base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA)
class RescheduleMaintenance(base.Command):
    """Reschedule a Cloud SQL instance's maintenance."""
    detailed_help = {'DESCRIPTION': textwrap.dedent("          {command} reschedules a Cloud SQL instance's maintenance.\n          "), 'EXAMPLES': textwrap.dedent('          To run maintenance on instance `my-instance` immediately, run:\n\n            $ {command} my-instance --reschedule-type=IMMEDIATE\n\n          To reschedule maintenance on instance `my-instance` to the next available window, run:\n\n            $ {command} my-instance --reschedule-type=NEXT_AVAILABLE_WINDOW\n\n          To reschedule maintenance on instance `my-instance` to 2019-11-07 at 4:00 am UTC, run:\n\n            $ {command} my-instance --reschedule-type=SPECIFIC_TIME --schedule-time=2019-11-07T04:00Z\n          ')}

    @staticmethod
    def Args(parser):
        """Args is called by calliope to gather arguments for this command.

    Args:
      parser: An argparse parser that you can use to add arguments that go on
        the command line after this command. Positional arguments are allowed.
    """
        flags.AddInstanceArgument(parser)
        flags.AddRescheduleType(parser)
        flags.AddScheduleTime(parser)

    def Run(self, args):
        """Runs the command to reschedule maintenance for a Cloud SQL instance."""
        client = api_util.SqlClient(api_util.API_VERSION_DEFAULT)
        return reschedule_maintenance_util.RunRescheduleMaintenanceCommand(args, client)